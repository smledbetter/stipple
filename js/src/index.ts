import { tableFromIPC } from "apache-arrow";
import { SCATTER_WGSL, LASSO_COMPUTE_WGSL, TAB10, BRAND_BLUE } from "./shaders";

interface Model {
  get<T = unknown>(key: string): T;
  set(key: string, value: unknown): void;
  save_changes(): void;
  on(event: string, handler: (...args: unknown[]) => void): void;
  send(
    content: unknown,
    callbacks?: unknown,
    buffers?: Array<ArrayBuffer | ArrayBufferView>,
  ): void;
}

interface RenderArgs {
  model: Model;
  el: HTMLElement;
}

function makeStatusEl(): HTMLDivElement {
  const el = document.createElement("div");
  el.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
  el.style.fontSize = "12px";
  el.style.padding = "6px 8px";
  el.style.background = "#f5f7fa";
  el.style.borderLeft = "3px solid #4a90e2";
  el.style.marginBottom = "6px";
  el.style.whiteSpace = "pre-wrap";
  return el;
}

function makeCanvasStack(width: number, height: number): {
  wrap: HTMLDivElement;
  gpuCanvas: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
} {
  const dpr = window.devicePixelRatio || 1;
  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.display = "inline-block";

  const gpuCanvas = document.createElement("canvas");
  gpuCanvas.width = Math.round(width * dpr);
  gpuCanvas.height = Math.round(height * dpr);
  gpuCanvas.style.width = `${width}px`;
  gpuCanvas.style.height = `${height}px`;
  gpuCanvas.style.display = "block";
  gpuCanvas.style.border = "1px solid #ddd";
  gpuCanvas.style.background = "#0d1117";
  gpuCanvas.style.cursor = "grab";
  gpuCanvas.style.touchAction = "none";

  const overlay = document.createElement("canvas");
  overlay.width = Math.round(width * dpr);
  overlay.height = Math.round(height * dpr);
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlay.style.position = "absolute";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.pointerEvents = "none";

  wrap.appendChild(gpuCanvas);
  wrap.appendChild(overlay);
  return { wrap, gpuCanvas, overlay };
}

function bufferToBytes(buf: ArrayBuffer | ArrayBufferView): Uint8Array {
  if (buf instanceof ArrayBuffer) return new Uint8Array(buf);
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

async function getAdapterInfo(adapter: GPUAdapter): Promise<string> {
  let info: Partial<GPUAdapterInfo> = {};
  try {
    const a = adapter as GPUAdapter & {
      requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
      info?: GPUAdapterInfo;
    };
    if (a.info) info = a.info;
    else if (typeof a.requestAdapterInfo === "function") info = await a.requestAdapterInfo();
  } catch {
    /* non-fatal */
  }
  return JSON.stringify(
    {
      vendor: info.vendor ?? null,
      architecture: info.architecture ?? null,
      device: info.device ?? null,
    },
    null,
    2,
  );
}

const MAX_POLYGON_VERTS = 512;

export default {
  async render({ model, el }: RenderArgs) {
    el.style.fontFamily = "system-ui, -apple-system, sans-serif";

    const status = makeStatusEl();
    const { wrap, gpuCanvas, overlay } = makeCanvasStack(640, 640);
    el.appendChild(status);
    el.appendChild(wrap);

    const overlayCtx = overlay.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    overlayCtx.scale(dpr, dpr);

    const log = (msg: string, isError = false) => {
      status.innerHTML = msg;
      status.style.color = isError ? "crimson" : "#222";
      status.style.background = isError ? "#fff5f5" : "#f5f7fa";
      status.style.borderLeftColor = isError ? "crimson" : "#4a90e2";
    };

    const report = (s: string, extra: { error?: string; adapter_info?: string } = {}) => {
      model.set("status", s);
      if ("error" in extra) model.set("error", extra.error ?? "");
      if ("adapter_info" in extra) model.set("adapter_info", extra.adapter_info ?? "");
      model.save_changes();
    };

    const fail = (s: string, msg: string, kind: string) => {
      log(`❌ ${kind}: ${msg}`, true);
      report(s, { error: msg });
    };

    if (!navigator.gpu) {
      fail("no-webgpu", "navigator.gpu is undefined", "WebGPU unsupported");
      return;
    }
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    if (!adapter) {
      fail("no-adapter", "requestAdapter returned null", "no GPU adapter");
      return;
    }
    const adapterInfo = await getAdapterInfo(adapter);
    const device = await adapter.requestDevice();

    const ctx = gpuCanvas.getContext("webgpu");
    if (!ctx) {
      fail("configure-error", "getContext('webgpu') returned null", "no webgpu context");
      return;
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: "premultiplied" });

    const writeBuf = (buf: GPUBuffer, offset: number, data: ArrayBufferView) =>
      device.queue.writeBuffer(buf, offset, data as unknown as ArrayBuffer);

    // -------- Render pipeline --------
    const renderShader = device.createShaderModule({ code: SCATTER_WGSL });
    const renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: renderShader, entryPoint: "vs" },
      fragment: {
        module: renderShader,
        entryPoint: "fs",
        targets: [
          {
            format,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    const ubo = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // -------- Compute pipeline (lasso hit-test) --------
    const computeShader = device.createShaderModule({ code: LASSO_COMPUTE_WGSL });
    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: computeShader, entryPoint: "cs" },
    });

    const polygonBuf = device.createBuffer({
      size: MAX_POLYGON_VERTS * 2 * 4, // vec2f
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const lassoUbo = device.createBuffer({
      size: 16, // u32 × 4 (n_points, poly_n, pad, pad)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const counterBuf = device.createBuffer({
      size: 4,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const counterStaging = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    type State = {
      n: number;
      posBuf: GPUBuffer;
      colorBuf: GPUBuffer;
      paletteBuf: GPUBuffer;
      paletteN: number;
      bindGroup: GPUBindGroup;
      computeBindGroup: GPUBindGroup;
      outIndices: GPUBuffer;
      outStaging: GPUBuffer;
      pointSize: number;
      viewTx: number;
      viewTy: number;
      viewSx: number;
      viewSy: number;
    };
    let state: State | null = null;

    function uploadState(
      positions: Float32Array,
      colorCodes: Uint32Array,
      paletteRGBA: Float32Array,
    ): State {
      const n = positions.length / 2;
      const paletteN = paletteRGBA.length / 4;

      const posBuf = device.createBuffer({
        size: positions.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeBuf(posBuf, 0, positions);

      const colorBuf = device.createBuffer({
        size: colorCodes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeBuf(colorBuf, 0, colorCodes);

      const paletteBuf = device.createBuffer({
        size: paletteRGBA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      writeBuf(paletteBuf, 0, paletteRGBA);

      const bindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: ubo } },
          { binding: 1, resource: { buffer: posBuf } },
          { binding: 2, resource: { buffer: colorBuf } },
          { binding: 3, resource: { buffer: paletteBuf } },
        ],
      });

      const outIndices = device.createBuffer({
        size: Math.max(4, n * 4),
        usage:
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const outStaging = device.createBuffer({
        size: Math.max(4, n * 4),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: posBuf } },
          { binding: 1, resource: { buffer: polygonBuf } },
          { binding: 2, resource: { buffer: counterBuf } },
          { binding: 3, resource: { buffer: outIndices } },
          { binding: 4, resource: { buffer: lassoUbo } },
        ],
      });

      const pointSize = n > 1_000_000 ? 1.2 : n > 100_000 ? 1.8 : n > 10_000 ? 2.5 : 5.0;

      return {
        n,
        posBuf,
        colorBuf,
        paletteBuf,
        paletteN,
        bindGroup,
        computeBindGroup,
        outIndices,
        outStaging,
        pointSize,
        viewTx: 0,
        viewTy: 0,
        viewSx: 1,
        viewSy: 1,
      };
    }

    function fitView(positions: Float32Array, s: State) {
      let xMin = Infinity;
      let xMax = -Infinity;
      let yMin = Infinity;
      let yMax = -Infinity;
      for (let i = 0; i < positions.length; i += 2) {
        const xi = positions[i];
        const yi = positions[i + 1];
        if (xi < xMin) xMin = xi;
        if (xi > xMax) xMax = xi;
        if (yi < yMin) yMin = yi;
        if (yi > yMax) yMax = yi;
      }
      const cx = (xMin + xMax) / 2;
      const cy = (yMin + yMax) / 2;
      const sx = xMax - xMin || 1;
      const sy = yMax - yMin || 1;
      const margin = 1.8;
      s.viewTx = cx;
      s.viewTy = cy;
      s.viewSx = margin / sx;
      s.viewSy = margin / sy;
    }

    function renderFrame() {
      if (!state) return;
      const data = new Float32Array([
        gpuCanvas.width,
        gpuCanvas.height,
        state.pointSize,
        state.paletteN,
        state.viewTx,
        state.viewTy,
        state.viewSx,
        state.viewSy,
      ]);
      writeBuf(ubo, 0, data);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: ctx!.getCurrentTexture().createView(),
            clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1.0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      pass.setPipeline(renderPipeline);
      pass.setBindGroup(0, state.bindGroup);
      pass.draw(6, state.n);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    let renderQueued = false;
    function requestRender() {
      if (renderQueued) return;
      renderQueued = true;
      requestAnimationFrame(() => {
        renderQueued = false;
        renderFrame();
      });
    }

    // -------- Coordinate transforms --------
    function screenToWorld(px: number, py: number): [number, number] {
      const rect = gpuCanvas.getBoundingClientRect();
      const sx = ((px - rect.left) / rect.width) * 2.0 - 1.0;
      const sy = -((py - rect.top) / rect.height) * 2.0 + 1.0;
      if (!state) return [sx, sy];
      const wx = sx / state.viewSx + state.viewTx;
      const wy = sy / state.viewSy + state.viewTy;
      return [wx, wy];
    }

    // -------- Lasso state --------
    type LassoMode = "idle" | "drawing" | "committed";
    let lassoMode: LassoMode = "idle";
    const lassoScreen: Array<[number, number]> = []; // CSS px, canvas-local
    const lassoWorld: Array<[number, number]> = [];

    function drawLassoOverlay() {
      overlayCtx.clearRect(0, 0, overlay.width / dpr, overlay.height / dpr);
      if (lassoScreen.length < 2) return;
      // Yellow while still drawing, soft green once committed.
      const isCommitted = lassoMode === "committed";
      overlayCtx.strokeStyle = isCommitted
        ? "rgba(110, 220, 140, 0.95)"
        : "rgba(255, 220, 80, 0.9)";
      overlayCtx.fillStyle = isCommitted
        ? "rgba(110, 220, 140, 0.10)"
        : "rgba(255, 220, 80, 0.10)";
      overlayCtx.lineWidth = isCommitted ? 1.5 : 1.5;
      overlayCtx.setLineDash(isCommitted ? [4, 3] : []);
      overlayCtx.beginPath();
      overlayCtx.moveTo(lassoScreen[0][0], lassoScreen[0][1]);
      for (let i = 1; i < lassoScreen.length; i++) {
        overlayCtx.lineTo(lassoScreen[i][0], lassoScreen[i][1]);
      }
      if (lassoScreen.length >= 3) {
        overlayCtx.closePath();
        overlayCtx.fill();
      }
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);
    }

    function clearLassoOverlay() {
      overlayCtx.clearRect(0, 0, overlay.width / dpr, overlay.height / dpr);
      lassoScreen.length = 0;
      lassoWorld.length = 0;
      lassoMode = "idle";
    }

    async function runLasso(): Promise<void> {
      if (!state || lassoWorld.length < 3) {
        clearLassoOverlay();
        return;
      }
      // Lock the drawn polygon in place with the "committed" styling so the
      // user can see exactly which region was selected.
      lassoMode = "committed";
      drawLassoOverlay();
      const tStart = performance.now();

      // Polygon → buffer
      const polyFlat = new Float32Array(lassoWorld.length * 2);
      for (let i = 0; i < lassoWorld.length; i++) {
        polyFlat[i * 2] = lassoWorld[i][0];
        polyFlat[i * 2 + 1] = lassoWorld[i][1];
      }
      writeBuf(polygonBuf, 0, polyFlat);
      writeBuf(
        lassoUbo,
        0,
        new Uint32Array([state.n, lassoWorld.length, 0, 0]),
      );
      writeBuf(counterBuf, 0, new Uint32Array([0]));

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, state.computeBindGroup);
      const wg = Math.ceil(state.n / 64);
      pass.dispatchWorkgroups(wg);
      pass.end();
      encoder.copyBufferToBuffer(counterBuf, 0, counterStaging, 0, 4);
      encoder.copyBufferToBuffer(state.outIndices, 0, state.outStaging, 0, state.n * 4);
      device.queue.submit([encoder.finish()]);

      await Promise.all([
        counterStaging.mapAsync(GPUMapMode.READ),
        state.outStaging.mapAsync(GPUMapMode.READ),
      ]);
      const count = new Uint32Array(counterStaging.getMappedRange())[0];
      const all = new Uint32Array(state.outStaging.getMappedRange());
      const slice = all.slice(0, count);
      counterStaging.unmap();
      state.outStaging.unmap();

      const ms = performance.now() - tStart;
      // Send to Python. Copy into a fresh ArrayBuffer to ensure ownership.
      const out = new Uint32Array(slice);
      model.send({ type: "selection", ms, count }, undefined, [out.buffer]);

      log(
        `✓ lasso: ${count.toLocaleString()} / ${state.n.toLocaleString()} selected · ${ms.toFixed(1)} ms · awaiting Python ack…\n` +
          `(shift+drag to lasso · plain drag to pan · wheel to zoom · re-run the next cell to inspect)`,
      );
      pendingAck = { count, ms };
    }

    let pendingAck: { count: number; ms: number } | null = null;
    model.on("change:selection_count", () => {
      if (!pendingAck || !state) return;
      const { count, ms } = pendingAck;
      pendingAck = null;
      log(
        `✓ lasso: ${count.toLocaleString()} / ${state.n.toLocaleString()} selected · gpu ${ms.toFixed(1)} ms · Python ack ✓\n` +
          `(shift+drag to lasso · plain drag to pan · wheel to zoom · re-run the next cell to inspect)`,
      );
    });

    // -------- Pan/zoom interaction (with shift-lasso branch) --------
    let panning = false;
    let panLastX = 0;
    let panLastY = 0;

    gpuCanvas.addEventListener("pointerdown", (e) => {
      if (!state) return;
      gpuCanvas.setPointerCapture(e.pointerId);
      if (e.shiftKey) {
        // Begin new lasso — clear any previously-committed outline first.
        clearLassoOverlay();
        lassoMode = "drawing";
        const rect = gpuCanvas.getBoundingClientRect();
        const lx = e.clientX - rect.left;
        const ly = e.clientY - rect.top;
        lassoScreen.push([lx, ly]);
        lassoWorld.push(screenToWorld(e.clientX, e.clientY));
        drawLassoOverlay();
      } else {
        panning = true;
        panLastX = e.clientX;
        panLastY = e.clientY;
        gpuCanvas.style.cursor = "grabbing";
        // Clear any committed lasso — pan would desync screen-coords from world.
        if (lassoMode === "committed") clearLassoOverlay();
      }
    });
    gpuCanvas.addEventListener("pointermove", (e) => {
      if (!state) return;
      if (lassoMode === "drawing") {
        const rect = gpuCanvas.getBoundingClientRect();
        const lx = e.clientX - rect.left;
        const ly = e.clientY - rect.top;
        const last = lassoScreen[lassoScreen.length - 1];
        const dx = lx - last[0];
        const dy = ly - last[1];
        if (
          dx * dx + dy * dy >= 9 /* >= 3 CSS-px */ &&
          lassoScreen.length < MAX_POLYGON_VERTS
        ) {
          lassoScreen.push([lx, ly]);
          lassoWorld.push(screenToWorld(e.clientX, e.clientY));
          drawLassoOverlay();
        }
      } else if (panning) {
        const dxPx = e.clientX - panLastX;
        const dyPx = e.clientY - panLastY;
        panLastX = e.clientX;
        panLastY = e.clientY;
        const rect = gpuCanvas.getBoundingClientRect();
        const dClipX = (dxPx / rect.width) * 2.0;
        const dClipY = -(dyPx / rect.height) * 2.0;
        state.viewTx -= dClipX / state.viewSx;
        state.viewTy -= dClipY / state.viewSy;
        requestRender();
      }
    });
    const endPointer = (e: PointerEvent) => {
      try {
        gpuCanvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      if (lassoMode === "drawing") {
        lassoMode = "idle";
        void runLasso();
      } else if (panning) {
        panning = false;
        gpuCanvas.style.cursor = "grab";
      }
    };
    gpuCanvas.addEventListener("pointerup", endPointer);
    gpuCanvas.addEventListener("pointercancel", endPointer);

    gpuCanvas.addEventListener(
      "wheel",
      (e) => {
        if (!state) return;
        e.preventDefault();
        const rect = gpuCanvas.getBoundingClientRect();
        const sx = ((e.clientX - rect.left) / rect.width) * 2.0 - 1.0;
        const sy = -((e.clientY - rect.top) / rect.height) * 2.0 + 1.0;
        const worldX = sx / state.viewSx + state.viewTx;
        const worldY = sy / state.viewSy + state.viewTy;
        const factor = Math.exp(-e.deltaY * 0.0015);
        state.viewSx *= factor;
        state.viewSy *= factor;
        state.viewTx = worldX - sx / state.viewSx;
        state.viewTy = worldY - sy / state.viewSy;
        if (lassoMode === "committed") clearLassoOverlay();
        requestRender();
      },
      { passive: false },
    );

    log(`⏳ Stipple — waiting for data…\nadapter: ${adapterInfo}`);

    function benchmarkFPS(frames = 30): Promise<{ avgMs: number; fps: number }> {
      return new Promise((resolve) => {
        const samples: number[] = [];
        let last = performance.now();
        let i = 0;
        const tick = (t: number) => {
          const dt = t - last;
          last = t;
          samples.push(dt);
          renderFrame();
          i++;
          if (i > frames + 2) {
            const valid = samples.slice(2);
            const avgMs = valid.reduce((a, b) => a + b, 0) / valid.length;
            resolve({ avgMs, fps: 1000 / avgMs });
          } else {
            requestAnimationFrame(tick);
          }
        };
        requestAnimationFrame(tick);
      });
    }

    model.on("msg:custom", ((..._args: unknown[]) => {
      const args = _args as [unknown, unknown[]];
      const msg = args[0] as { type?: string } | undefined;
      const buffers = args[1] as Array<ArrayBuffer | ArrayBufferView> | undefined;
      if (!msg || msg.type !== "data") return;
      if (!buffers || buffers.length === 0) {
        fail("decode-error", "data msg without buffer", "no buffer");
        return;
      }
      void (async () => {
        try {
          const tStart = performance.now();
          const bytes = bufferToBytes(buffers[0]);
          const tDecode0 = performance.now();
          const table = tableFromIPC(bytes);
          const x = table.getChild("x")!.toArray() as Float32Array;
          const y = table.getChild("y")!.toArray() as Float32Array;
          const colorCol = table.getChild("color");
          const tDecode1 = performance.now();

          const n = x.length;
          const positions = new Float32Array(n * 2);
          for (let i = 0; i < n; i++) {
            positions[i * 2] = x[i];
            positions[i * 2 + 1] = y[i];
          }

          let colorCodes: Uint32Array;
          let paletteRGBA: Float32Array;
          const cats = model.get<unknown[]>("color_categories") ?? [];
          if (colorCol && cats.length > 0) {
            const raw = colorCol.toArray();
            colorCodes =
              raw instanceof Uint32Array ? raw : new Uint32Array(raw as ArrayLike<number>);
            const k = Math.max(1, cats.length);
            paletteRGBA = new Float32Array(k * 4);
            for (let i = 0; i < k; i++) {
              const rgb = TAB10[i % TAB10.length];
              paletteRGBA[i * 4 + 0] = rgb[0];
              paletteRGBA[i * 4 + 1] = rgb[1];
              paletteRGBA[i * 4 + 2] = rgb[2];
              paletteRGBA[i * 4 + 3] = 1.0;
            }
          } else {
            colorCodes = new Uint32Array(n);
            paletteRGBA = new Float32Array([
              BRAND_BLUE[0],
              BRAND_BLUE[1],
              BRAND_BLUE[2],
              1.0,
            ]);
          }

          const tUpload0 = performance.now();
          state = uploadState(positions, colorCodes, paletteRGBA);
          fitView(positions, state);
          const tUpload1 = performance.now();

          renderFrame();
          const tRender = performance.now() - tUpload1;

          model.set("rows_received", n);
          model.set("bytes_received", bytes.byteLength);
          model.set("status", "data-rendered");
          model.save_changes();

          const { avgMs, fps } = await benchmarkFPS(30);
          model.set("avg_frame_ms", avgMs);
          model.set("last_fps", fps);
          model.save_changes();

          const mb = (bytes.byteLength / (1024 * 1024)).toFixed(2);
          const totalMs = (performance.now() - tStart).toFixed(0);
          log(
            `✓ Stipple — ${n.toLocaleString()} rows · ${cats.length || 1} class${(cats.length || 1) === 1 ? "" : "es"}\n` +
              `Arrow IPC: ${mb} MB · decode: ${(tDecode1 - tDecode0).toFixed(1)} ms · upload: ${(tUpload1 - tUpload0).toFixed(1)} ms · first render: ${tRender.toFixed(1)} ms\n` +
              `FPS: ${fps.toFixed(1)} (frame ${avgMs.toFixed(2)} ms) · shift+drag to lasso · drag to pan · wheel to zoom\n` +
              `total: ${totalMs} ms · adapter: ${adapterInfo}`,
          );
        } catch (e) {
          fail("decode-error", String(e), "decode/render threw");
        }
      })();
    }) as (...args: unknown[]) => void);

    model.set("client_ready", true);
    model.save_changes();
    report("ready", { adapter_info: adapterInfo });
  },
};
