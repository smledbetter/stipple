import { tableFromIPC } from "apache-arrow";
import {
  SCATTER_WGSL,
  LASSO_COMPUTE_WGSL,
  MASK_CLEAR_WGSL,
  GRID_COUNT_WGSL,
  GRID_SCATTER_WGSL,
  HOVER_QUERY_WGSL,
  HOVER_GRID_CELL_N,
  DENSITY_BUILD_WGSL,
  DENSITY_RENDER_WGSL,
  DENSITY_BIN_N,
  MAX_REDUCE_WGSL,
  BRAND_BLUE,
} from "./shaders";

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
  tooltip: HTMLDivElement;
} {
  const dpr = window.devicePixelRatio || 1;
  const wrap = document.createElement("div");
  wrap.style.position = "relative";
  wrap.style.display = "inline-block";
  wrap.setAttribute("data-stipple-role", "wrap");

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
  gpuCanvas.setAttribute("data-stipple-role", "render");

  const overlay = document.createElement("canvas");
  overlay.width = Math.round(width * dpr);
  overlay.height = Math.round(height * dpr);
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlay.style.position = "absolute";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.pointerEvents = "none";
  overlay.setAttribute("data-stipple-role", "overlay");

  const tooltip = document.createElement("div");
  tooltip.style.position = "absolute";
  tooltip.style.pointerEvents = "none";
  tooltip.style.background = "rgba(20, 22, 28, 0.92)";
  tooltip.style.color = "#fff";
  tooltip.style.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  tooltip.style.padding = "4px 7px";
  tooltip.style.borderRadius = "3px";
  tooltip.style.whiteSpace = "pre";
  tooltip.style.display = "none";
  tooltip.style.zIndex = "3";
  tooltip.setAttribute("data-stipple-role", "tooltip");

  wrap.appendChild(gpuCanvas);
  wrap.appendChild(overlay);
  wrap.appendChild(tooltip);
  return { wrap, gpuCanvas, overlay, tooltip };
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
    const { wrap, gpuCanvas, overlay, tooltip } = makeCanvasStack(640, 640);
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
    // Opt into the adapter's maxima for storage-buffer + total-buffer size.
    // Default limits cap storage bindings at 128 MiB and total buffer size
    // at 256 MiB — both are hit well below 25M points. We cap requested
    // values to whatever the adapter actually supports.
    const aLimits = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: aLimits.maxStorageBufferBindingSize,
        maxBufferSize: aLimits.maxBufferSize,
      },
    });

    const ctx = gpuCanvas.getContext("webgpu");
    if (!ctx) {
      fail("configure-error", "getContext('webgpu') returned null", "no webgpu context");
      return;
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: "premultiplied" });

    const writeBuf = (buf: GPUBuffer, offset: number, data: ArrayBufferView) =>
      device.queue.writeBuffer(buf, offset, data as unknown as ArrayBuffer);

    // dispatchWorkgroups is capped at 65535 per dimension. For bulk-scan
    // kernels (workgroup_size 256), 1D dispatch tops out at ~16.8M threads.
    // Use 2D dispatch beyond that — shaders compute the flat index from
    // (gid.x + gid.y * num_workgroups.x * 256u).
    const MAX_DISPATCH_DIM = 65535;
    const BULK_WG = 256;
    function dispatchN(pass: GPUComputePassEncoder, n: number) {
      const groups = Math.ceil(n / BULK_WG);
      if (groups <= MAX_DISPATCH_DIM) {
        pass.dispatchWorkgroups(groups);
      } else {
        const gx = MAX_DISPATCH_DIM;
        const gy = Math.ceil(groups / MAX_DISPATCH_DIM);
        pass.dispatchWorkgroups(gx, gy);
      }
    }

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
      // 8 f32 of payload (viewport, point_size, palette_n, view_translate,
      // view_scale) + selection_dim + has_selection + 2 pad floats = 48 bytes.
      // Uniform buffers must be a multiple of 16 bytes; WGSL pads the struct
      // up to that boundary, and the host buffer must match.
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // -------- Compute pipeline (lasso hit-test) --------
    const computeShader = device.createShaderModule({ code: LASSO_COMPUTE_WGSL });
    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: computeShader, entryPoint: "cs" },
    });

    // -------- Mask-clear compute pipeline --------
    // Zeroes the selection mask before each lasso. Cheap (~1–2 ms at 10M).
    const maskClearShader = device.createShaderModule({ code: MASK_CLEAR_WGSL });
    const maskClearPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: maskClearShader, entryPoint: "cs" },
    });

    // -------- Hover-index compute pipelines (build + query) --------
    const gridCountShader = device.createShaderModule({ code: GRID_COUNT_WGSL });
    const gridCountPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: gridCountShader, entryPoint: "cs" },
    });
    const gridScatterShader = device.createShaderModule({ code: GRID_SCATTER_WGSL });
    const gridScatterPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: gridScatterShader, entryPoint: "cs" },
    });
    const hoverQueryShader = device.createShaderModule({ code: HOVER_QUERY_WGSL });
    const hoverQueryPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: hoverQueryShader, entryPoint: "cs" },
    });

    // -------- Max-reduce pipeline (used by progressive density renders) ---
    const maxReduceShader = device.createShaderModule({ code: MAX_REDUCE_WGSL });
    const maxReducePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: maxReduceShader, entryPoint: "cs" },
    });

    const CELL_N = HOVER_GRID_CELL_N;
    const CELL_TOTAL = CELL_N * CELL_N;
    const BIN_N = DENSITY_BIN_N;
    const BIN_TOTAL = BIN_N * BIN_N;

    // -------- Density-mode compute + render pipelines --------
    const densityBuildShader = device.createShaderModule({ code: DENSITY_BUILD_WGSL });
    const densityBuildPipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: densityBuildShader, entryPoint: "cs" },
    });
    const densityRenderShader = device.createShaderModule({ code: DENSITY_RENDER_WGSL });
    const densityRenderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: densityRenderShader, entryPoint: "vs" },
      fragment: {
        module: densityRenderShader,
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
    const densityRenderUbo = device.createBuffer({
      // viewport(8) + bin_n(4) + log_max(4) + view_translate(8) + view_scale(8)
      // + origin(8) + inv_cell(8) + palette_n(4) + pad(12) = 64 bytes.
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Reusable buffers for hover-query result (8 bytes: dist² u32, idx u32).
    const queryOut = device.createBuffer({
      size: 8,
      usage:
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const queryStaging = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const queryUbo = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
      maskClearBindGroup: GPUBindGroup;
      maskClearUbo: GPUBuffer;
      selectionMask: GPUBuffer;
      outIndices: GPUBuffer;
      outStaging: GPUBuffer;
      pointSize: number;
      viewTx: number;
      viewTy: number;
      viewSx: number;
      viewSy: number;
      hasSelection: boolean;
      // Hover index (built async after uploadState returns)
      positionsCpu: Float32Array;
      gridReady: boolean;
      gridOrigin: [number, number];
      gridInvCell: [number, number];
      cellStartBuf: GPUBuffer;
      cellCountBuf: GPUBuffer;
      pointIdsBuf: GPUBuffer;
      hoverQueryBindGroup: GPUBindGroup;
      // Density mode (built async via buildDensityGrid)
      densityReady: boolean;
      densityOrigin: [number, number];
      densityInvCell: [number, number];
      binCountBuf: GPUBuffer;
      densityRenderBindGroup: GPUBindGroup;
      densityLogMax: number;
      // The mode this state was *uploaded* for. We lock this at upload time
      // so an interactive `render_mode` change can't accidentally request
      // operations against a buffer we already discarded.
      mode: "scatter" | "density" | "density-only";
    };
    let state: State | null = null;

    function uploadState(
      positions: Float32Array,
      colorCodes: Uint32Array,
      paletteRGBA: Float32Array,
    ): State {
      const n = positions.length / 2;
      const paletteN = paletteRGBA.length / 4;
      const positionsCpu = positions;

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

      // Per-point selection mask (1 = in current selection, 0 = not).
      // Initialised to zero by the GPU when created.
      const selectionMask = device.createBuffer({
        size: Math.max(4, n * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const bindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: ubo } },
          { binding: 1, resource: { buffer: posBuf } },
          { binding: 2, resource: { buffer: colorBuf } },
          { binding: 3, resource: { buffer: paletteBuf } },
          { binding: 4, resource: { buffer: selectionMask } },
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
          { binding: 5, resource: { buffer: selectionMask } },
        ],
      });

      const maskClearUbo = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      writeBuf(maskClearUbo, 0, new Uint32Array([n, 0, 0, 0]));
      const maskClearBindGroup = device.createBindGroup({
        layout: maskClearPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: selectionMask } },
          { binding: 1, resource: { buffer: maskClearUbo } },
        ],
      });

      const pointSize = n > 1_000_000 ? 1.2 : n > 100_000 ? 1.8 : n > 10_000 ? 2.5 : 5.0;

      // -------- Hover-index buffers (data filled by buildHoverGrid below) --
      const cellStartBuf = device.createBuffer({
        size: CELL_TOTAL * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      const cellCountBuf = device.createBuffer({
        size: CELL_TOTAL * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      const pointIdsBuf = device.createBuffer({
        size: Math.max(4, n * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });

      const hoverQueryBindGroup = device.createBindGroup({
        layout: hoverQueryPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: posBuf } },
          { binding: 1, resource: { buffer: cellStartBuf } },
          { binding: 2, resource: { buffer: cellCountBuf } },
          { binding: 3, resource: { buffer: pointIdsBuf } },
          { binding: 4, resource: { buffer: queryUbo } },
          { binding: 5, resource: { buffer: queryOut } },
        ],
      });

      // Density-mode bin grid: BIN_N × BIN_N u32 counts.
      const binCountBuf = device.createBuffer({
        size: BIN_TOTAL * 4,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      const densityRenderBindGroup = device.createBindGroup({
        layout: densityRenderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: densityRenderUbo } },
          { binding: 1, resource: { buffer: binCountBuf } },
          { binding: 2, resource: { buffer: paletteBuf } },
        ],
      });

      return {
        n,
        posBuf,
        colorBuf,
        paletteBuf,
        paletteN,
        bindGroup,
        computeBindGroup,
        maskClearBindGroup,
        maskClearUbo,
        selectionMask,
        outIndices,
        outStaging,
        pointSize,
        viewTx: 0,
        viewTy: 0,
        viewSx: 1,
        viewSy: 1,
        hasSelection: false,
        positionsCpu,
        gridReady: false,
        gridOrigin: [0, 0],
        gridInvCell: [1, 1],
        cellStartBuf,
        cellCountBuf,
        pointIdsBuf,
        hoverQueryBindGroup,
        densityReady: false,
        densityOrigin: [0, 0],
        densityInvCell: [1, 1],
        binCountBuf,
        densityRenderBindGroup,
        densityLogMax: 1,
        mode: "scatter",
      };
    }

    // Build the hover index grid. Two-pass (count + scatter) with a CPU
    // prefix-sum sandwiched in the middle for the per-cell offsets. Runs
    // once per upload; cost scales with n and is dwarfed by the IPC decode
    // for the realistic sizes (1M–10M).
    async function buildHoverGrid(s: State): Promise<void> {
      // World bbox from CPU positions (cheap, single pass).
      let xMin = Infinity;
      let xMax = -Infinity;
      let yMin = Infinity;
      let yMax = -Infinity;
      const p = s.positionsCpu;
      for (let i = 0; i < p.length; i += 2) {
        const xi = p[i];
        const yi = p[i + 1];
        if (xi < xMin) xMin = xi;
        if (xi > xMax) xMax = xi;
        if (yi < yMin) yMin = yi;
        if (yi > yMax) yMax = yi;
      }
      // Pad bbox slightly so points at the edge don't fall outside cell 0/N-1
      // after floating-point rounding.
      const padX = ((xMax - xMin) || 1) * 1e-4;
      const padY = ((yMax - yMin) || 1) * 1e-4;
      xMin -= padX;
      xMax += padX;
      yMin -= padY;
      yMax += padY;
      const cellW = (xMax - xMin) / CELL_N;
      const cellH = (yMax - yMin) / CELL_N;
      const invCellX = cellW > 0 ? 1 / cellW : 1;
      const invCellY = cellH > 0 ? 1 / cellH : 1;
      s.gridOrigin = [xMin, yMin];
      s.gridInvCell = [invCellX, invCellY];

      const gridUbo = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const gridHeader = new ArrayBuffer(32);
      new Uint32Array(gridHeader, 0, 4).set([s.n, CELL_N, 0, 0]);
      new Float32Array(gridHeader, 16, 4).set([xMin, yMin, invCellX, invCellY]);
      writeBuf(gridUbo, 0, new Uint8Array(gridHeader));

      // Scratch buffer for atomic counter during scatter pass.
      const cellCursorBuf = device.createBuffer({
        size: CELL_TOTAL * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const cellCountStaging = device.createBuffer({
        size: CELL_TOTAL * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      // Zero count + cursor buffers, then dispatch the count pass.
      const zeros = new Uint32Array(CELL_TOTAL);
      writeBuf(s.cellCountBuf, 0, zeros);
      writeBuf(cellCursorBuf, 0, zeros);

      const countBindGroup = device.createBindGroup({
        layout: gridCountPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: s.posBuf } },
          { binding: 1, resource: { buffer: s.cellCountBuf } },
          { binding: 2, resource: { buffer: gridUbo } },
        ],
      });

      const enc1 = device.createCommandEncoder();
      const pass1 = enc1.beginComputePass();
      pass1.setPipeline(gridCountPipeline);
      pass1.setBindGroup(0, countBindGroup);
      dispatchN(pass1, s.n);
      pass1.end();
      enc1.copyBufferToBuffer(s.cellCountBuf, 0, cellCountStaging, 0, CELL_TOTAL * 4);
      device.queue.submit([enc1.finish()]);

      await cellCountStaging.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(cellCountStaging.getMappedRange()).slice();
      cellCountStaging.unmap();
      cellCountStaging.destroy();

      // CPU prefix sum → cell_start.
      const starts = new Uint32Array(CELL_TOTAL);
      let acc = 0;
      for (let i = 0; i < CELL_TOTAL; i++) {
        starts[i] = acc;
        acc += counts[i];
      }
      writeBuf(s.cellStartBuf, 0, starts);

      const scatterBindGroup = device.createBindGroup({
        layout: gridScatterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: s.posBuf } },
          { binding: 1, resource: { buffer: s.cellStartBuf } },
          { binding: 2, resource: { buffer: cellCursorBuf } },
          { binding: 3, resource: { buffer: s.pointIdsBuf } },
          { binding: 4, resource: { buffer: gridUbo } },
        ],
      });

      const enc2 = device.createCommandEncoder();
      const pass2 = enc2.beginComputePass();
      pass2.setPipeline(gridScatterPipeline);
      pass2.setBindGroup(0, scatterBindGroup);
      dispatchN(pass2, s.n);
      pass2.end();
      device.queue.submit([enc2.finish()]);

      cellCursorBuf.destroy();
      gridUbo.destroy();
      s.gridReady = true;
    }

    // Build the BIN_N × BIN_N density grid for render_mode="density". Single
    // compute pass over the points; CPU readback finds the max bin count to
    // calibrate the log-color map.
    async function buildDensityGrid(s: State): Promise<void> {
      // World bbox (re-derive here so buildDensityGrid stays independent of
      // buildHoverGrid's ordering).
      let xMin = Infinity;
      let xMax = -Infinity;
      let yMin = Infinity;
      let yMax = -Infinity;
      const p = s.positionsCpu;
      for (let i = 0; i < p.length; i += 2) {
        const xi = p[i];
        const yi = p[i + 1];
        if (xi < xMin) xMin = xi;
        if (xi > xMax) xMax = xi;
        if (yi < yMin) yMin = yi;
        if (yi > yMax) yMax = yi;
      }
      const padX = ((xMax - xMin) || 1) * 1e-4;
      const padY = ((yMax - yMin) || 1) * 1e-4;
      xMin -= padX;
      xMax += padX;
      yMin -= padY;
      yMax += padY;
      const cellW = (xMax - xMin) / BIN_N;
      const cellH = (yMax - yMin) / BIN_N;
      const invX = cellW > 0 ? 1 / cellW : 1;
      const invY = cellH > 0 ? 1 / cellH : 1;
      s.densityOrigin = [xMin, yMin];
      s.densityInvCell = [invX, invY];

      // Build-pass uniform buffer.
      const buildUbo = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const buildHeader = new ArrayBuffer(32);
      new Uint32Array(buildHeader, 0, 4).set([s.n, BIN_N, 0, 0]);
      new Float32Array(buildHeader, 16, 4).set([xMin, yMin, invX, invY]);
      writeBuf(buildUbo, 0, new Uint8Array(buildHeader));

      // Zero the bin count buffer.
      const zeros = new Uint32Array(BIN_TOTAL);
      writeBuf(s.binCountBuf, 0, zeros);

      const buildBindGroup = device.createBindGroup({
        layout: densityBuildPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: s.posBuf } },
          { binding: 1, resource: { buffer: s.binCountBuf } },
          { binding: 2, resource: { buffer: buildUbo } },
        ],
      });

      const staging = device.createBuffer({
        size: BIN_TOTAL * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(densityBuildPipeline);
      pass.setBindGroup(0, buildBindGroup);
      dispatchN(pass, s.n);
      pass.end();
      enc.copyBufferToBuffer(s.binCountBuf, 0, staging, 0, BIN_TOTAL * 4);
      device.queue.submit([enc.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const counts = new Uint32Array(staging.getMappedRange());
      let maxC = 0;
      for (let i = 0; i < counts.length; i++) {
        if (counts[i] > maxC) maxC = counts[i];
      }
      staging.unmap();
      staging.destroy();
      buildUbo.destroy();

      s.densityLogMax = Math.log(maxC + 1);
      s.densityReady = true;
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

    function activeRenderMode(): "scatter" | "density" | "density-only" {
      if (!state) return "scatter";
      // density-only is locked at upload time — positions buffer is gone,
      // there's no way to render scatter anymore.
      if (state.mode === "density-only") return "density-only";
      const requested = (model.get<string>("render_mode") || "scatter") as
        | "scatter"
        | "density"
        | "density-only";
      if (requested === "density-only") {
        // The trait was changed AFTER an upload in a different mode; can't
        // retroactively drop positions. Just render density.
        return state.densityReady ? "density" : "scatter";
      }
      if (requested === "density" && !state.densityReady) return "scatter";
      return requested;
    }

    function renderFrame() {
      if (!state) return;
      const mode = activeRenderMode();
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

      if (mode === "density" || mode === "density-only") {
        const header = new ArrayBuffer(64);
        new Float32Array(header, 0, 2).set([gpuCanvas.width, gpuCanvas.height]);
        new Uint32Array(header, 8, 1)[0] = BIN_N;
        new Float32Array(header, 12, 1)[0] = state.densityLogMax;
        new Float32Array(header, 16, 4).set([
          state.viewTx,
          state.viewTy,
          state.viewSx,
          state.viewSy,
        ]);
        new Float32Array(header, 32, 4).set([
          state.densityOrigin[0],
          state.densityOrigin[1],
          state.densityInvCell[0],
          state.densityInvCell[1],
        ]);
        new Uint32Array(header, 48, 1)[0] = state.paletteN;
        writeBuf(densityRenderUbo, 0, new Uint8Array(header));

        pass.setPipeline(densityRenderPipeline);
        pass.setBindGroup(0, state.densityRenderBindGroup);
        pass.draw(3, 1);
      } else {
        const dim = Math.min(
          1.0,
          Math.max(0.0, model.get<number>("selection_dim") ?? 0.4),
        );
        const data = new Float32Array([
          gpuCanvas.width,
          gpuCanvas.height,
          state.pointSize,
          state.paletteN,
          state.viewTx,
          state.viewTy,
          state.viewSx,
          state.viewSy,
          dim,
          state.hasSelection ? 1.0 : 0.0,
          0.0,
          0.0, // pad to 48-byte uniform buffer (16-aligned)
        ]);
        writeBuf(ubo, 0, data);
        pass.setPipeline(renderPipeline);
        pass.setBindGroup(0, state.bindGroup);
        pass.draw(6, state.n);
      }

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

      // First: zero the per-point selection mask from any prior lasso.
      const clearPass = encoder.beginComputePass();
      clearPass.setPipeline(maskClearPipeline);
      clearPass.setBindGroup(0, state.maskClearBindGroup);
      dispatchN(clearPass, state.n);
      clearPass.end();

      // Then: lasso hit-test. Writes to out_indices + selection_mask.
      const pass = encoder.beginComputePass();
      pass.setPipeline(computePipeline);
      pass.setBindGroup(0, state.computeBindGroup);
      dispatchN(pass, state.n);
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
      state.hasSelection = true;
      // Burst several RAFs so the post-lasso frame is presented before the
      // page settles — a single RAF-debounced render sometimes never reaches
      // the compositor, leaving stale pixels in screenshots.
      let bursts = 5;
      const burst = () => {
        if (bursts-- <= 0) return;
        renderFrame();
        requestAnimationFrame(burst);
      };
      burst();

      // Send the selection indices to Python. The ipywidget comm channel
      // sits on a JupyterLab websocket with a default 10 MiB max-message-
      // size — selections above ~2.6M indices (10.4 MB) drop silently.
      // We chunk anything larger into 4 MiB pieces and reassemble Python-
      // side. Small selections still use the single-shot path.
      const out = new Uint32Array(slice); // fresh-buffer copy for ownership
      const SEL_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB
      if (out.byteLength <= SEL_CHUNK_BYTES) {
        model.send({ type: "selection", ms, count }, undefined, [out.buffer]);
      } else {
        const u32PerChunk = SEL_CHUNK_BYTES / 4;
        const nChunks = Math.ceil(count / u32PerChunk);
        selGenCounter += 1;
        const gen = selGenCounter;
        model.send(
          { type: "selection_start", gen, ms, count, n_chunks: nChunks },
          undefined,
          [],
        );
        for (let i = 0; i < nChunks; i++) {
          const a = i * u32PerChunk;
          const b = Math.min(a + u32PerChunk, count);
          // ArrayBuffer.slice copies — necessary so each chunk has its own
          // backing buffer and the comm serializer doesn't try to ship the
          // full underlying 40 MB region for every message.
          const chunkBuf = out.buffer.slice(a * 4, b * 4);
          model.send({ type: "selection_chunk", gen, i }, undefined, [chunkBuf]);
        }
        model.send({ type: "selection_finalize", gen }, undefined, []);
      }

      log(
        `✓ lasso: ${count.toLocaleString()} / ${state.n.toLocaleString()} selected · ${ms.toFixed(1)} ms · awaiting Python ack…\n` +
          `(shift+drag to lasso · plain drag to pan · wheel to zoom · re-run the next cell to inspect)`,
      );
      pendingAck = { count, ms };
    }
    let selGenCounter = 0;

    model.on("change:render_mode", () => {
      if (!state) return;
      // Hide tooltip on mode switch — semantics differ between modes.
      tooltip.style.display = "none";
      renderFrame();
    });

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
        // Density-only mode has no positions buffer — lasso is meaningless.
        // Silently swallow the shift-drag so it doesn't half-draw an overlay.
        if (state.mode === "density-only") return;
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

    // -------- Hover tooltip --------
    let queryInflight = false;
    let lastHoverPx: { x: number; y: number } | null = null;
    let hoverShouldShow = false;
    const hideTooltip = () => {
      tooltip.style.display = "none";
      hoverShouldShow = false;
    };

    function queryHover(clientX: number, clientY: number) {
      if (!state || !state.gridReady) return;
      if (lassoMode !== "idle" || panning) return;
      const m = activeRenderMode();
      if (m === "density" || m === "density-only") {
        hideTooltip();
        return;
      }
      const rect = gpuCanvas.getBoundingClientRect();
      const lx = clientX - rect.left;
      const ly = clientY - rect.top;
      if (lx < 0 || ly < 0 || lx > rect.width || ly > rect.height) {
        hideTooltip();
        return;
      }
      lastHoverPx = { x: lx, y: ly };
      if (queryInflight) return; // a later mousemove will trigger another
      queryInflight = true;
      hoverShouldShow = true;

      const [wx, wy] = screenToWorld(clientX, clientY);
      const header = new ArrayBuffer(48);
      new Float32Array(header, 0, 2).set([wx, wy]);
      new Uint32Array(header, 8, 2).set([CELL_N, 1]); // radius_cells = 1 → 3x3
      new Float32Array(header, 16, 4).set([
        state.gridOrigin[0],
        state.gridOrigin[1],
        state.gridInvCell[0],
        state.gridInvCell[1],
      ]);
      writeBuf(queryUbo, 0, new Uint8Array(header));

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(hoverQueryPipeline);
      pass.setBindGroup(0, state.hoverQueryBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(queryOut, 0, queryStaging, 0, 8);
      device.queue.submit([enc.finish()]);

      // Pixel threshold → world distance². Use the current view scale.
      const thresholdPx = 14;
      const worldPerPxX = 2 / rect.width / state.viewSx;
      const worldPerPxY = 2 / rect.height / state.viewSy;
      const worldThresh = Math.max(
        thresholdPx * Math.abs(worldPerPxX),
        thresholdPx * Math.abs(worldPerPxY),
      );
      const threshSq = worldThresh * worldThresh;

      void queryStaging.mapAsync(GPUMapMode.READ).then(() => {
        const r = new Uint32Array(queryStaging.getMappedRange()).slice();
        queryStaging.unmap();
        queryInflight = false;
        if (!state || !hoverShouldShow || !lastHoverPx) return;
        const d2 = new Float32Array(r.buffer, 0, 1)[0];
        const idx = r[1];
        if (idx === 0xffffffff || !isFinite(d2) || d2 > threshSq) {
          hideTooltip();
          return;
        }
        const px = state.positionsCpu[idx * 2];
        const py = state.positionsCpu[idx * 2 + 1];
        const lines = [
          `row: ${idx.toLocaleString()}`,
          `x: ${px.toFixed(4)}`,
          `y: ${py.toFixed(4)}`,
        ];
        tooltip.textContent = lines.join("\n");
        tooltip.style.display = "block";
        // Position slightly offset from cursor so it doesn't sit under the pointer.
        const tx = Math.min(rect.width - 12, lastHoverPx.x + 12);
        const ty = Math.min(rect.height - 12, lastHoverPx.y + 12);
        tooltip.style.left = `${tx}px`;
        tooltip.style.top = `${ty}px`;
      });
    }

    gpuCanvas.addEventListener("pointermove", (e) => {
      if (!state) return;
      if (lassoMode !== "idle" || panning) {
        hideTooltip();
        return;
      }
      queryHover(e.clientX, e.clientY);
    });
    gpuCanvas.addEventListener("pointerleave", hideTooltip);

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
        hideTooltip();
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

    // -------- Chunked stream state (P5.1b) --------
    // Holds metadata + GPU resources during a multi-message chunked upload.
    // Single-shot 'data' messages bypass this entirely.
    type StreamState = {
      gen: number;
      mode: "scatter" | "density" | "density-only";
      n: number;
      nChunks: number;
      chunkN: number;
      bbox: [number, number, number, number]; // [xmin, xmax, ymin, ymax]
      received: number;
      paletteRGBA: Float32Array;
      tStart: number;
      bytesReceived: number;
      // For scatter / density modes: accumulate the full positions + colors
      // on the JS side, then construct State via uploadState at finalize.
      positions?: Float32Array;
      colorCodes?: Uint32Array;
      // For density-only: per-chunk GPU upload + density build, then drop.
      transientPosBuf?: GPUBuffer;
      densityBuildUbo?: GPUBuffer;
      binCountBuf?: GPUBuffer;
      // Density-only render plumbing (allocated lazily at first chunk).
      densityRenderBindGroup?: GPUBindGroup;
      paletteBuf?: GPUBuffer;
      // Progressive rendering (density-only): fast GPU max-reduce so we can
      // update the colormap without a CPU readback over the whole bin grid.
      progressiveK: number;
      progressiveCount: number;
      reduceInflight: boolean;
      maxReduceUbo?: GPUBuffer;
      maxReduceOutBuf?: GPUBuffer;
      maxReduceStaging?: GPUBuffer;
      maxReduceBindGroup?: GPUBindGroup;
    };
    let stream: StreamState | null = null;
    // Accumulator for chunked update_color() messages — recolor sends
    // raw uint32 codes (no Arrow round-trip) because every byte is data.
    // `chunksRemaining` lets finalize detect dropped chunks and abort
    // rather than commit a partial recolor that paints a slice of points
    // in stale colors.
    let colorUpdateStream: {
      gen: number;
      n: number;
      codes: Uint32Array;
      chunksRemaining: number;
    } | null = null;
    // Generation counter so back-to-back update_color() calls don't race;
    // each Python send carries a monotonically-increasing `gen`. We drop
    // any message whose `gen` is older than the latest accepted one.
    let latestColorGen = 0;

    async function handleDataStart(
      msg: {
        gen: number;
        n: number;
        n_chunks: number;
        chunk_n: number;
        bbox: number[];
        render_mode: "scatter" | "density" | "density-only";
      },
      paletteBuffer: ArrayBuffer | ArrayBufferView,
    ): Promise<void> {
      // Abort any prior in-flight stream.
      if (stream) {
        stream.transientPosBuf?.destroy();
        stream.densityBuildUbo?.destroy();
        stream.binCountBuf?.destroy();
        stream.paletteBuf?.destroy();
        stream.maxReduceUbo?.destroy();
        stream.maxReduceOutBuf?.destroy();
        stream.maxReduceStaging?.destroy();
        stream = null;
      }
      const palBytes = bufferToBytes(paletteBuffer);
      const paletteRGBA = new Float32Array(
        new Float32Array(
          palBytes.buffer,
          palBytes.byteOffset,
          palBytes.byteLength / 4,
        ),
      );

      stream = {
        gen: msg.gen,
        mode: msg.render_mode,
        n: msg.n,
        nChunks: msg.n_chunks,
        chunkN: msg.chunk_n,
        bbox: [msg.bbox[0], msg.bbox[1], msg.bbox[2], msg.bbox[3]],
        received: 0,
        paletteRGBA,
        tStart: performance.now(),
        bytesReceived: 0,
        // Render progressively every K chunks. Target ~8 progressive frames
        // total so 25-40 chunk streams get healthy mid-load feedback while
        // small streams (≤8 chunks) update on every chunk.
        progressiveK: Math.max(1, Math.ceil(msg.n_chunks / 8)),
        progressiveCount: 0,
        reduceInflight: false,
      };
      // Reset the synced trait so multiple uploads don't accumulate.
      model.set("progressive_renders", 0);
      model.save_changes();

      if (msg.render_mode === "density-only") {
        // Allocate only the buffers we'll actually keep past finalize:
        // bin_count grid + palette + render UBO/bind group. Plus a transient
        // per-chunk positions buffer used to feed the density build kernel.
        stream.transientPosBuf = device.createBuffer({
          size: Math.max(8, msg.chunk_n * 8),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        stream.densityBuildUbo = device.createBuffer({
          size: 32,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        stream.binCountBuf = device.createBuffer({
          size: BIN_TOTAL * 4,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });
        // Zero the bin grid.
        writeBuf(stream.binCountBuf, 0, new Uint32Array(BIN_TOTAL));
        stream.paletteBuf = device.createBuffer({
          size: paletteRGBA.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        writeBuf(stream.paletteBuf, 0, paletteRGBA);
        stream.densityRenderBindGroup = device.createBindGroup({
          layout: densityRenderPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: densityRenderUbo } },
            { binding: 1, resource: { buffer: stream.binCountBuf } },
            { binding: 2, resource: { buffer: stream.paletteBuf } },
          ],
        });

        // Compute origin + inv_cell once from the Python-supplied bbox so
        // every chunk's build pass uses identical bin boundaries.
        const [xMin, xMax, yMin, yMax] = stream.bbox;
        const padX = ((xMax - xMin) || 1) * 1e-4;
        const padY = ((yMax - yMin) || 1) * 1e-4;
        const origin: [number, number] = [xMin - padX, yMin - padY];
        const cellW = (xMax - xMin + 2 * padX) / BIN_N;
        const cellH = (yMax - yMin + 2 * padY) / BIN_N;
        const invCell: [number, number] = [
          cellW > 0 ? 1 / cellW : 1,
          cellH > 0 ? 1 / cellH : 1,
        ];
        // Density-only writes a minimal state object so the existing render
        // path works. The unused fields point at the transient buffer too.
        const stub = stream.transientPosBuf!;
        state = {
          n: msg.n,
          posBuf: stub,
          colorBuf: stub,
          paletteBuf: stream.paletteBuf!,
          paletteN: paletteRGBA.length / 4,
          bindGroup: device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: ubo } },
              { binding: 1, resource: { buffer: stub } },
              { binding: 2, resource: { buffer: stub } },
              { binding: 3, resource: { buffer: stream.paletteBuf! } },
              { binding: 4, resource: { buffer: stub } },
            ],
          }),
          computeBindGroup: null as unknown as GPUBindGroup,
          maskClearBindGroup: null as unknown as GPUBindGroup,
          maskClearUbo: stub,
          selectionMask: stub,
          outIndices: stub,
          outStaging: stub,
          pointSize: 1.2,
          viewTx: 0,
          viewTy: 0,
          viewSx: 1,
          viewSy: 1,
          hasSelection: false,
          positionsCpu: new Float32Array(0),
          gridReady: false,
          gridOrigin: [0, 0],
          gridInvCell: [1, 1],
          cellStartBuf: stub,
          cellCountBuf: stub,
          pointIdsBuf: stub,
          hoverQueryBindGroup: null as unknown as GPUBindGroup,
          densityReady: false, // flipped at finalize
          densityOrigin: origin,
          densityInvCell: invCell,
          binCountBuf: stream.binCountBuf!,
          densityRenderBindGroup: stream.densityRenderBindGroup!,
          densityLogMax: 1,
          mode: "density-only",
        };
        // Use bbox to set the view directly (no positions-iteration).
        const cx = (xMin + xMax) / 2;
        const cy = (yMin + yMax) / 2;
        const sx = (xMax - xMin) || 1;
        const sy = (yMax - yMin) || 1;
        const margin = 1.8;
        state.viewTx = cx;
        state.viewTy = cy;
        state.viewSx = margin / sx;
        state.viewSy = margin / sy;

        // Pre-fill the build UBO header (it doesn't change per chunk except
        // n_points, which we patch in place).
        const hdr = new ArrayBuffer(32);
        new Uint32Array(hdr, 0, 4).set([0, BIN_N, 0, 0]); // n_points filled per chunk
        new Float32Array(hdr, 16, 4).set([origin[0], origin[1], invCell[0], invCell[1]]);
        writeBuf(stream.densityBuildUbo!, 0, new Uint8Array(hdr));

        // Allocate max-reduce buffers for progressive renders.
        stream.maxReduceUbo = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        writeBuf(stream.maxReduceUbo, 0, new Uint32Array([BIN_TOTAL, 0, 0, 0]));
        stream.maxReduceOutBuf = device.createBuffer({
          size: 4,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        });
        stream.maxReduceStaging = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        stream.maxReduceBindGroup = device.createBindGroup({
          layout: maxReducePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stream.binCountBuf! } },
            { binding: 1, resource: { buffer: stream.maxReduceOutBuf } },
            { binding: 2, resource: { buffer: stream.maxReduceUbo } },
          ],
        });
      } else {
        // scatter / density: accumulate the full data CPU-side, then build
        // state once at finalize using uploadState (same as single-shot).
        stream.positions = new Float32Array(msg.n * 2);
        stream.colorCodes = new Uint32Array(msg.n);
      }

      log(
        `⏳ Stipple — loading 0 / ${msg.n_chunks} chunks · ${msg.n.toLocaleString()} rows · mode=${msg.render_mode}\n` +
          `adapter: ${adapterInfo}`,
      );
    }

    // Progressive density render: run the GPU max-reduce on the current
    // bin_count grid, read back the single u32 max, update log_max, render.
    // Fire-and-forget during chunked streams (skip if a prior reduce is
    // still in flight); awaited at finalize so the last frame is accurate.
    async function progressiveRender(s: StreamState): Promise<void> {
      if (s.reduceInflight) return;
      if (!s.maxReduceBindGroup || !s.maxReduceOutBuf || !s.maxReduceStaging) {
        return;
      }
      s.reduceInflight = true;
      try {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(maxReducePipeline);
        pass.setBindGroup(0, s.maxReduceBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(s.maxReduceOutBuf, 0, s.maxReduceStaging, 0, 4);
        device.queue.submit([enc.finish()]);

        await s.maxReduceStaging.mapAsync(GPUMapMode.READ);
        const maxVal = new Uint32Array(s.maxReduceStaging.getMappedRange())[0];
        s.maxReduceStaging.unmap();

        if (state && state.mode === "density-only") {
          state.densityLogMax = Math.log(maxVal + 1);
          state.densityReady = true;
          renderFrame();
          s.progressiveCount += 1;
        }
      } finally {
        s.reduceInflight = false;
      }
    }

    async function handleDataChunk(
      msg: { gen: number; i: number },
      ipcBuffer: ArrayBuffer | ArrayBufferView,
    ): Promise<void> {
      if (!stream || msg.gen !== stream.gen) return;
      const ipcBytes = bufferToBytes(ipcBuffer);
      stream.bytesReceived += ipcBytes.byteLength;
      const table = tableFromIPC(ipcBytes);
      const x = table.getChild("x")!.toArray() as Float32Array;
      const y = table.getChild("y")!.toArray() as Float32Array;
      const colorCol = table.getChild("color");
      const chunkN = x.length;
      const chunkPositions = new Float32Array(chunkN * 2);
      for (let i = 0; i < chunkN; i++) {
        chunkPositions[i * 2] = x[i];
        chunkPositions[i * 2 + 1] = y[i];
      }
      let chunkCodes: Uint32Array;
      if (colorCol) {
        const raw = colorCol.toArray();
        chunkCodes =
          raw instanceof Uint32Array ? raw : new Uint32Array(raw as ArrayLike<number>);
      } else {
        chunkCodes = new Uint32Array(chunkN);
      }

      const offsetPoints = msg.i * stream.chunkN;

      if (stream.mode === "density-only") {
        // Upload this chunk's positions to the transient buffer, then run
        // the density build over [0..chunkN). atomicAdds accumulate into
        // the persistent bin_count buffer.
        writeBuf(stream.transientPosBuf!, 0, chunkPositions);
        // Patch n_points in the build UBO.
        writeBuf(stream.densityBuildUbo!, 0, new Uint32Array([chunkN]));
        const buildBindGroup = device.createBindGroup({
          layout: densityBuildPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stream.transientPosBuf! } },
            { binding: 1, resource: { buffer: stream.binCountBuf! } },
            { binding: 2, resource: { buffer: stream.densityBuildUbo! } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(densityBuildPipeline);
        pass.setBindGroup(0, buildBindGroup);
        dispatchN(pass, chunkN);
        pass.end();
        device.queue.submit([enc.finish()]);
      } else {
        // Stash into the JS-side full-size arrays; uploadState consumes
        // these at finalize.
        stream.positions!.set(chunkPositions, offsetPoints * 2);
        stream.colorCodes!.set(chunkCodes, offsetPoints);
      }

      stream.received += 1;
      log(
        `⏳ Stipple — loading ${stream.received} / ${stream.nChunks} chunks · ${stream.n.toLocaleString()} rows · mode=${stream.mode}`,
      );

      // Progressive render: in density-only mode, every K chunks the canvas
      // visualizes the cumulative bin grid so the user sees clusters emerge
      // rather than waiting on a dark canvas. handleDataFinalize fires the
      // last render, so skip when this is the final chunk.
      if (
        stream.mode === "density-only" &&
        stream.received < stream.nChunks &&
        stream.received % stream.progressiveK === 0
      ) {
        void progressiveRender(stream);
      }
    }

    async function handleDataFinalize(msg: { gen: number }): Promise<void> {
      if (!stream || msg.gen !== stream.gen) return;
      const s = stream;
      stream = null;

      if (s.mode === "density-only") {
        // Final render: GPU max-reduce → small u32 readback → render. If a
        // progressive reduce is already in flight, wait it out so we don't
        // submit two concurrent reduces against the same buffers.
        while (s.reduceInflight) {
          await new Promise((r) => setTimeout(r, 4));
        }
        await progressiveRender(s);
        s.densityBuildUbo?.destroy();
        s.transientPosBuf?.destroy();
        s.maxReduceUbo?.destroy();
        s.maxReduceOutBuf?.destroy();
        s.maxReduceStaging?.destroy();
      } else {
        // scatter / density: assemble State from the accumulated arrays.
        const positions = s.positions!;
        const colorCodes = s.colorCodes!;
        state = uploadState(positions, colorCodes, s.paletteRGBA);
        state.mode = s.mode;
        fitView(positions, state);
        renderFrame();
        // Background-build hover grid (scatter only) and density grid.
        if (s.mode === "scatter") {
          void buildHoverGrid(state).catch((err) => {
            console.warn("[stipple] hover grid build failed:", err);
          });
        }
        void buildDensityGrid(state)
          .then(() => {
            if (state && state.mode === "density") renderFrame();
          })
          .catch((err) => {
            console.warn("[stipple] density grid build failed:", err);
          });
      }

      const totalMs = (performance.now() - s.tStart).toFixed(0);
      const mb = (s.bytesReceived / (1024 * 1024)).toFixed(2);
      model.set("rows_received", s.n);
      model.set("bytes_received", s.bytesReceived);
      model.set("status", "data-rendered");
      model.set("progressive_renders", s.progressiveCount);
      model.save_changes();

      const { avgMs, fps } = await benchmarkFPS(30);
      model.set("avg_frame_ms", avgMs);
      model.set("last_fps", fps);
      model.save_changes();

      log(
        `✓ Stipple — ${s.n.toLocaleString()} rows · mode=${s.mode}\n` +
          `Arrow IPC: ${mb} MB across ${s.nChunks} chunks · ${totalMs} ms wall\n` +
          `FPS: ${fps.toFixed(1)} (frame ${avgMs.toFixed(2)} ms) · ` +
          (s.mode === "density-only"
            ? "lasso + hover disabled (density-only)\n"
            : "shift+drag to lasso · drag to pan · wheel to zoom\n") +
          `adapter: ${adapterInfo}`,
      );
    }

    model.on("msg:custom", ((..._args: unknown[]) => {
      const args = _args as [unknown, unknown[]];
      const msg = args[0] as { type?: string; [k: string]: unknown } | undefined;
      const buffers = args[1] as Array<ArrayBuffer | ArrayBufferView> | undefined;
      if (!msg) return;
      if (msg.type === "data_start") {
        if (!buffers || buffers.length === 0) {
          fail("decode-error", "data_start without palette buffer", "no buffer");
          return;
        }
        void handleDataStart(
          msg as unknown as Parameters<typeof handleDataStart>[0],
          buffers[0],
        ).catch((e) => fail("decode-error", String(e), "data_start threw"));
        return;
      }
      if (msg.type === "data_chunk") {
        if (!buffers || buffers.length === 0) return;
        void handleDataChunk(
          msg as unknown as Parameters<typeof handleDataChunk>[0],
          buffers[0],
        ).catch((e) => fail("decode-error", String(e), "data_chunk threw"));
        return;
      }
      if (msg.type === "data_finalize") {
        void handleDataFinalize(
          msg as unknown as Parameters<typeof handleDataFinalize>[0],
        ).catch((e) => fail("decode-error", String(e), "data_finalize threw"));
        return;
      }
      if (msg.type === "color_update") {
        const m = msg as unknown as { gen: number; n: number };
        if (!state) {
          // eslint-disable-next-line no-console
          console.warn("[stipple] color_update arrived before data load");
          return;
        }
        if (m.gen < latestColorGen) return;  // stale relative to in-flight chunked update
        latestColorGen = m.gen;
        if (!buffers || buffers.length === 0) return;
        const bytes = bufferToBytes(buffers[0]);
        const codes = new Uint32Array(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength / 4,
        );
        writeBuf(state.colorBuf, 0, codes);
        requestRender();
        return;
      }
      if (msg.type === "color_update_start") {
        const m = msg as unknown as { gen: number; n: number; n_chunks: number };
        if (m.gen < latestColorGen) return;
        colorUpdateStream = {
          gen: m.gen,
          n: m.n,
          codes: new Uint32Array(m.n),
          chunksRemaining: m.n_chunks,
        };
        return;
      }
      if (msg.type === "color_update_chunk") {
        const m = msg as unknown as { gen: number; a: number };
        if (
          !colorUpdateStream ||
          colorUpdateStream.gen !== m.gen ||
          !buffers ||
          buffers.length === 0
        ) {
          return;
        }
        const bytes = bufferToBytes(buffers[0]);
        const chunk = new Uint32Array(
          bytes.buffer,
          bytes.byteOffset,
          bytes.byteLength / 4,
        );
        colorUpdateStream.codes.set(chunk, m.a);
        colorUpdateStream.chunksRemaining -= 1;
        return;
      }
      if (msg.type === "color_update_finalize") {
        const m = msg as unknown as { gen: number };
        const s = colorUpdateStream;
        colorUpdateStream = null;
        if (!state || !s || s.gen !== m.gen) return;
        if (s.chunksRemaining !== 0) {
          // eslint-disable-next-line no-console
          console.warn(
            `[stipple] dropping recolor — ${s.chunksRemaining} chunks never arrived`,
          );
          return;
        }
        latestColorGen = s.gen;
        writeBuf(state.colorBuf, 0, s.codes);
        requestRender();
        return;
      }
      if (msg.type !== "data") return;
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
          if (colorCol) {
            const raw = colorCol.toArray();
            colorCodes =
              raw instanceof Uint32Array ? raw : new Uint32Array(raw as ArrayLike<number>);
          } else {
            colorCodes = new Uint32Array(n);
          }

          // Palette ships from Python as buffers[1] (Kx4 float32 RGBA).
          // Falls back to brand blue if a previous-version Python omits it.
          let paletteRGBA: Float32Array;
          if (buffers.length > 1 && buffers[1]) {
            const palBytes = bufferToBytes(buffers[1]);
            const f32 = new Float32Array(
              palBytes.buffer,
              palBytes.byteOffset,
              palBytes.byteLength / 4,
            );
            // Copy so we don't share the backing ArrayBuffer with the comm message.
            paletteRGBA = new Float32Array(f32);
          } else {
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
          // Build the hover grid asynchronously — tooltip stays disabled
          // until gridReady flips true, which is a few ms after first render.
          void buildHoverGrid(state).catch((err) => {
            // eslint-disable-next-line no-console
            console.warn("[stipple] hover grid build failed:", err);
          });
          // Build the density grid asynchronously. If the user requested
          // density mode, kick a re-render when it's ready.
          void buildDensityGrid(state)
            .then(() => {
              if (model.get<string>("render_mode") === "density") renderFrame();
            })
            .catch((err) => {
              // eslint-disable-next-line no-console
              console.warn("[stipple] density grid build failed:", err);
            });

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
          const cats = model.get<unknown[]>("color_categories") ?? [];
          const range = model.get<number[]>("color_range") ?? [];
          let colorLine: string;
          if (range.length === 2) {
            colorLine = `continuous color [${range[0].toFixed(3)}, ${range[1].toFixed(3)}]`;
          } else if (cats.length > 0) {
            colorLine = `${cats.length} class${cats.length === 1 ? "" : "es"}`;
          } else {
            colorLine = "single color";
          }
          log(
            `✓ Stipple — ${n.toLocaleString()} rows · ${colorLine}\n` +
              `Arrow IPC: ${mb} MB · decode: ${(tDecode1 - tDecode0).toFixed(1)} ms · upload: ${(tUpload1 - tUpload0).toFixed(1)} ms · first render: ${tRender.toFixed(1)} ms\n` +
              `FPS: ${fps.toFixed(1)} (frame ${avgMs.toFixed(2)} ms) · shift+drag to lasso · drag to pan · wheel to zoom · re-run the next cell to inspect\n` +
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
