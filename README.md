# Stipple

A WebGPU scatter widget for Jupyter that renders **up to 600 million points in
a browser tab** — interactive at 10 M with 60 FPS pan/zoom and shift-drag
lasso → numpy round-trip in milliseconds, density-rendered all the way to
TPC-H `lineitem` at SF=100. No tiling, no server, no levels-of-detail.

```python
from stipple import Stipple

w = Stipple(x=xs, y=ys, color=labels)   # 10M points, scatter mode
w

# ── user shift-drags a lasso around a cluster ──

w.selected_indices    # np.ndarray[uint32]   — back in Python instantly
```

Stipple is the open-source WebGPU implementation of the **RasterScan** index
introduced in the CIDR 2026 paper
[*"Raster is Faster: Rethinking Ray Tracing in Database Indexing"*](https://www.vldb.org/cidrdb/papers/2026/p18-doraiswamy.pdf)
by Doraiswamy & Haritsa, packaged as an
[anywidget](https://anywidget.dev/) so it works in JupyterLab, Jupyter Notebook,
VS Code, and Colab with `pip install stipple`.

![Paint by lasso on 10M FineWeb embeddings](docs/paint-by-lasso-demo.gif)

***This is real time*** — no time compression, no skipped frames
([full-quality MP4](docs/paint-by-lasso-demo.mp4)). 10 M FineWeb-Edu
documents load, the user shift-drags a lasso, a Python cell uses
`w.selected_indices` to compute a Gaussian falloff from the lassoed centroid,
and `w.update_color()` re-shades all 10 M points in one GPU buffer swap.

---

## Why a new scatter widget?

If you work with embeddings, the standard interactive options each have a wall
you keep running into:

|                         | Scale (points) | Lasso → `numpy` in kernel | Recolor without re-upload |
| ----------------------- | -------------- | ------------------------- | ------------------------- |
| TensorBoard Projector   | ~500 K         | no (download CSV)         | no                        |
| Plotly / bqplot / bokeh | ~100 K – 1 M   | yes, but rate-limited     | full re-render            |
| datashader              | billions       | rasters → no per-point    | full re-aggregation       |
| deepscatter             | ~5–10 M        | partial (WebGL)           | encoding object reset     |
| Nomic Atlas             | 100 M+         | cloud round-trip          | server call               |
| **Stipple**             | **10 M scatter / 600 M density** | **`w.selected_indices` in ms (≤10 M)** | **`w.update_color(arr)` — one GPU buffer swap** |

The line that nothing else does cleanly is the bottom one: **lasso a region,
run a Python cell that uses the indices, hand the result back, watch every
point re-shade in the same notebook tab.** That feedback loop is the thing
worth getting fast.

## What you get

```bash
pip install stipple
```

- **Scatter mode** — one anti-aliased disk per row. Up to ~10 M points at 60 FPS
  on an M-series Mac.
- **Density mode** — 1024² log-binned heatmap. Scales to 100 M+ (and 600 M+ at
  SF=100 TPC-H, see below).
- **Lasso compute on the GPU** — shift-drag traces a polygon; a compute shader
  evaluates point-in-polygon against the full positions buffer and writes the
  matching row indices back to a staging buffer. The Python kernel sees
  `w.selected_indices` populated within a few milliseconds of mouse-up.
- **Runtime recolor** — `w.update_color(arr)` re-quantizes a 1-D score vector
  against the palette fixed at init and ships a uint32 codes buffer back to the
  GPU. Every point re-shades in a single buffer swap.
- **Real numpy / Arrow / pandas / polars input** — `Stipple(df, x="col_a",
  y="col_b", color="col_c")` accepts pandas, polars, pyarrow tables, or plain
  arrays.

## Quickstart — the "paint by lasso" loop

```python
import numpy as np
from stipple import Stipple

# 10M points (here just synthetic — substitute your embedding).
N = 10_000_000
rng = np.random.default_rng(0)
xs = rng.normal(0.5, 0.15, N).astype(np.float32)
ys = rng.normal(0.5, 0.15, N).astype(np.float32)

w = Stipple(x=xs, y=ys, color=np.zeros(N), color_kind="continuous")
w
```

After shift-dragging a region:

```python
# w.selected_indices is now populated (uint32 ndarray).
idx = w.selected_indices
seed_x = float(xs[idx].mean())
seed_y = float(ys[idx].mean())

# Recolor by Gaussian closeness to the lassoed centroid.
sigma = 0.05
sims = np.exp(-((xs - seed_x)**2 + (ys - seed_y)**2) / (2 * sigma**2))
w.update_color(sims)        # 10M points repaint in one GPU upload
```

The lassoed region lights up bright; semantic neighbors fade in by similarity;
the rest goes dark. The recolor runs in milliseconds — the limit is the
Python-side Gaussian, not the GPU. See
[`examples/demo_fineweb_paint.ipynb`](examples/demo_fineweb_paint.ipynb) for the
demo run on 10 M real FineWeb-Edu documents.

## Performance

Measurements on M4 MacBook Air, Chromium WebGPU/Metal:

| Workload                                                  | Stipple        | Reference                |
| --------------------------------------------------------- | -------------- | ------------------------ |
| 10 M scatter, pan/zoom/lasso                              | **60 FPS**     | —                        |
| Lasso → `selected_indices` (10 M, ~3 M selected)          | **~5 ms** GPU compute, ~80 ms wall incl. comm | — |
| `update_color()` recolor (10 M, full repaint)             | **~30 ms** wall | — |
| TPC-H `lineitem` drilldown at SF=25 (150 M rows)          | **3.3 ms**     | DuckDB-WASM: 2.8 s (**842×**) |
| Synthesized TPC-H `lineitem` at SF=100 (600 M rows)       | loads + queries | DuckDB-WASM: fails (V8 ArrayBuffer cap at 26 s) |

The big numbers are the rasterization-as-range-query trick at work, not loop
unrolling.

## Academic provenance

Stipple is a faithful WebGPU adaptation of **RasterScan**, the index introduced
in:

> Harish Doraiswamy and Jayant R. Haritsa.
> ["Raster is Faster: Rethinking Ray Tracing in Database Indexing."](https://www.vldb.org/cidrdb/papers/2026/p18-doraiswamy.pdf)
> *Conference on Innovative Data Systems Research (CIDR), 2026.*

The paper's central observation: a 2-D range query is structurally the same as
drawing the query rectangle into a 2-D texture-bucketed index — the GPU's
rasterizer already does the comparison work, you just need to interpret the
result. The paper shows this beats RT-cores-based indexing (RTIndeX, RayDB) and
on classic axis-aligned range queries beats CPU B-trees by orders of magnitude.

Stipple keeps the paper's substrate — 2-D texture-bucketed index, prefix-sum +
bin-assign build, two-pass range query — and adapts the original Vulkan
reference to WebGPU:

| Paper construct                           | Stipple WebGPU equivalent                                |
| ----------------------------------------- | -------------------------------------------------------- |
| Geometry-shader bin assignment            | Compute-shader workgroup + atomic counter                |
| Fragment-shader texture atomics           | Storage-buffer atomic-add via WGSL `atomicAdd`           |
| Render-target tile counts                 | Storage buffer + readback to host                        |

On top of that substrate, Stipple adds three things the paper doesn't have:

1. **Arbitrary convex polygon predicates.** The paper covers axis-aligned
   range queries; Stipple runs a polygon point-in-polygon test in a compute
   shader against the bin-resident candidate set, so any shift-drag works as a
   predicate.
2. **In-kernel `numpy` round-trip.** The matching row indices come back over
   the ipywidgets comm channel (chunked to clear the 10 MiB JupyterLab cap
   when needed) so the same notebook cell that issued the query gets a numpy
   array out.
3. **Runtime recolor without rebuilding the index.** Positions stay on the
   GPU; a separate uint32 color-codes buffer can be swapped via
   `update_color(arr)`, so a Python-computed score vector becomes a heatmap in
   one upload.

## How it works

**Python side** (`src/stipple/widget.py`) is an `anywidget.AnyWidget` subclass.
Input data is staged into Arrow IPC streams and shipped over the ipywidgets
comm channel — single-shot for small N, chunked into 4 M-row Arrow tables for
larger loads. The lasso reply path is symmetric: when the selection-indices
buffer exceeds the 10 MiB websocket cap, the JS side splits the uint32 buffer
into 4 MiB chunks; Python reassembles in `_on_custom_msg` before bumping
`selection_count` to notify observers.

**JS side** (`js/src/index.ts`, built to `src/stipple/_static/index.js` via
Vite) is a WebGPU pipeline: a compute shader builds the per-cell histogram and
prefix-sum table on first load, a render pipeline draws the points (instanced
quads in scatter mode, log-binned density grid in density mode), and a second
compute shader runs the lasso polygon predicate against the positions buffer
when shift-drag finishes. Camera matrix lives in a small uniform buffer so pan
and zoom don't touch GPU memory.

`update_color()` swaps just the per-point `colorBuf` and triggers a single
`requestRender()` — the index and positions stay untouched on the GPU, which
is why a recolor at 10 M points is sub-frame.

## Examples

| Notebook                                                              | What it shows                                         |
| --------------------------------------------------------------------- | ----------------------------------------------------- |
| [`demo_fineweb_paint.ipynb`](examples/demo_fineweb_paint.ipynb)       | The "paint by lasso" loop on 10 M FineWeb embeddings  |
| [`demo_fineweb_10m.ipynb`](examples/demo_fineweb_10m.ipynb)           | Lasso a cluster → inspect the actual document texts   |
| [`dedup.ipynb`](examples/dedup.ipynb)                                 | MinHash-LSH dedup visualization with planted duplicates |
| [`embeddings.ipynb`](examples/embeddings.ipynb)                       | Walk-through of pandas / polars / Arrow DataFrame input |
| [`p5_density_smoke.ipynb`](examples/p5_density_smoke.ipynb)           | Density mode at 25 M and 100 M points                 |
| [`p51b_tpch_smoke.ipynb`](examples/p51b_tpch_smoke.ipynb)             | TPC-H `lineitem` at SF=100 (600 M rows) in a browser  |

## Status

v0.1.0 — first public release. The render path, lasso, chunked transport,
runtime recolor, and DataFrame ingest are all green at the smoke gates in
[`examples/`](examples/). The API is unlikely to change shape but trait names
and keyword arguments may be refined before 1.0. Bug reports welcome on the
issue tracker.

## Development setup

```bash
git clone https://github.com/smledbetter/stipple
cd stipple
uv sync --extra dev --extra examples
cd js && npm install && npm run build && cd ..

# Run the example notebooks
uv run jupyter lab examples/

# Run the headless probes (Playwright + WebGPU/Metal)
uv run python scripts/probe_paint.py
```

Tests rely on a real WebGPU adapter, so Playwright Chromium is launched with
`--enable-unsafe-webgpu --use-angle=metal`.

## Citation

If Stipple is useful in your work, please cite both the paper and the package:

```bibtex
@inproceedings{doraiswamy2026raster,
  title     = {Raster is Faster: Rethinking Ray Tracing in Database Indexing},
  author    = {Doraiswamy, Harish and Haritsa, Jayant R.},
  booktitle = {Conference on Innovative Data Systems Research (CIDR)},
  year      = {2026},
  url       = {https://www.vldb.org/cidrdb/papers/2026/p18-doraiswamy.pdf},
}

@software{stipple2026,
  title  = {Stipple: a WebGPU implementation of RasterScan as a Jupyter widget},
  author = {Ledbetter, Steve},
  year   = {2026},
  url    = {https://github.com/smledbetter/stipple},
}
```

## License

MIT — see [LICENSE](LICENSE).
