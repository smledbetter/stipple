# Changelog

All notable changes to Stipple are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
pre-1.0 means trait names and keyword arguments may still be refined.

## [0.1.0] — 2026-05-15

First public release on PyPI. WebGPU implementation of the CIDR 2026
RasterScan index, packaged as an [anywidget](https://anywidget.dev/) so
it runs in JupyterLab, Jupyter Notebook, VS Code, and Colab with
`pip install stipple`.

### Added

- `Stipple(x=, y=, color=)` Jupyter widget in two render modes:
  - `"scatter"` — one anti-aliased disk per row, 60 FPS up to 10 M points
  - `"density"` — 1024² log-binned heatmap, scales to 600 M+ rows (verified
    on synthesized TPC-H `lineitem` at SF=100)
- **GPU lasso → numpy.** Shift-drag traces a polygon; a WebGPU compute
  shader evaluates point-in-polygon over the full positions buffer and
  writes matching row indices back. `w.selected_indices` is a
  `numpy.ndarray[uint32]` populated within ~80 ms of mouse-up at 10 M
  scale (≈5 ms GPU compute, the rest is comm-channel round-trip).
- **`update_color(arr)` — runtime recolor without rebuilding the index.**
  Quantizes a 1-D score vector against the palette fixed at init, ships
  a uint32 codes buffer to the GPU, and re-renders in a single buffer
  swap (~30 ms wall at 10 M). The whole point of the "paint by lasso"
  loop — see [`examples/demo_fineweb_paint.ipynb`](examples/demo_fineweb_paint.ipynb).
- **Chunked Arrow IPC transport** for input data (4 M-row Arrow tables
  per chunk) and **chunked uint32 selection reply** (4 MiB chunks) to
  clear the 10 MiB JupyterLab websocket cap. Both paths are
  generation-counter-tagged so back-to-back updates can't race.
- **Pandas / polars / pyarrow / numpy ingest.** `Stipple(df, x="col_a",
  y="col_b", color="col_c")` accepts any of the four.
- **Demo recording.** Real-time MP4 of the paint-by-lasso workflow on
  10 M FineWeb-Edu documents, attached to the GitHub release and
  embedded as a GIF in the README. Reproducible via
  `scripts/record_fineweb_demo.py`.
- **End-to-end probe.** `scripts/probe_paint.py` exercises the full
  load → lasso → recolor loop headlessly via Playwright + WebGPU/Metal
  and asserts a ≥3.0 mean-abs-pixel diff between pre/post-recolor
  screenshots.

### Academic provenance

Faithful WebGPU adaptation of the index introduced in
[Doraiswamy & Haritsa, "Raster is Faster: Rethinking Ray Tracing in
Database Indexing"](https://www.vldb.org/cidrdb/papers/2026/p18-doraiswamy.pdf)
(CIDR 2026), extended with three things the paper doesn't have:

1. Arbitrary convex polygon predicates (paper covers axis-aligned only)
2. In-kernel `numpy` round-trip from the same notebook cell
3. Runtime recolor via `update_color()` without rebuilding the index

[0.1.0]: https://github.com/smledbetter/stipple/releases/tag/v0.1.0
