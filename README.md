# Stipple

WebGPU scatter widget for Jupyter. Render millions of points, lasso a region, get the row indices back in Python.

```python
from stipple import Stipple
w = Stipple(df, x="umap_0", y="umap_1", color="label")
w
# ... user lassos in widget ...
w.selected_indices  # numpy array
```

## Status

Pre-release. Gate ladder:

- [x] **G0** — Package skeleton + 100-point WebGPU render
- [ ] **G1** — Arrow-over-comm transport (1M rows)
- [ ] **G2** — 1M scatter with pan/zoom + discrete color
- [ ] **G3** — 10M scale via chunked buffers
- [ ] **G4** — Polygon lasso → `selected_indices`
- [ ] **G5** — Example notebooks + cross-env test matrix
- [ ] **G6** — Wheel packaging

## Dev setup

```bash
uv sync --extra dev
cd js && npm install && npm run build && cd ..
uv run jupyter lab examples/
```

## Architecture

- Python: `anywidget.AnyWidget` subclass with `traitlets`. Arrow IPC over the comm channel for tensor/DataFrame transport.
- JS: Vite-built ESM bundle at `src/stipple/_static/index.js`. WebGPU scatter pipeline with instanced unit quads.

See `Projects/Research/Stipple/JUPYTER-WIDGET-RESEARCH.md` in the upstream Thinking vault for design rationale.

## License

MIT
