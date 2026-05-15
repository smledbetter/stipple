# Stipple env support matrix

The automated probes in `scripts/probe_g*.py` cover **JupyterLab on M4
Chromium with WebGPU flags** end-to-end. The matrix below is the manual
verification grid for the remaining environments. Fill in as tested.

| Environment | Status | Notes |
|---|---|---|
| JupyterLab / Chromium (probe) | ✓ verified | `scripts/probe_g4.py` green; FPS 60+ at 10M |
| JupyterLab / real Chrome 134+ | _pending_ | Open `examples/embeddings.ipynb`; check the lasso |
| JupyterLab / real Safari 26+ | _pending_ | Safari 26 ships WebGPU on Apple Silicon |
| VSCode notebook | _pending_ | Webview is Chromium-based; expected to work |
| Google Colab | _pending_ | Cross-origin output iframe — may fail; not load-bearing |
| JupyterHub | _pending_ | `anywidget` must be installed in the kernel env, not just the launch env |

## How to fill in a row manually

```bash
# Install the dev extras (jupyterlab + sklearn + ipywidgets, etc.)
uv sync --extra dev --extra examples
# Build the JS bundle if you haven't yet
cd js && npm install && npm run build && cd ..
# Launch
uv run jupyter lab examples/embeddings.ipynb
```

Then in the browser:
1. Run all cells
2. Shift+drag in the widget canvas
3. Run the inspection cell — `w.selected_indices` should be populated

For Safari, navigate manually to the JupyterLab URL printed in the terminal.

For Colab: `pip install stipple` inside a Colab cell, then upload the example
notebook. Per the upstream research note this row is the load-bearing one
for "covers most ML researchers"; treat any failure as expected and document
the symptom.

For VSCode: open the example notebook directly in VSCode (Python +
Jupyter extensions installed), select the `.venv` interpreter, run cells.

## Known caveats

- `widget model not found` console errors during initial widget setup are
  benign — JupyterLab's manager probes for the model before the comm
  channel completes registration. The final view is properly connected,
  proven by the bidirectional trait sync.
- WebGPU is gated behind a permissions banner in some Chrome
  configurations; if no adapter is found, check `chrome://flags` for
  `#enable-unsafe-webgpu` and Linux users may also need `#enable-vulkan`.
- The widget requires HTTPS or `localhost` — WebGPU is blocked on plain
  HTTP origins outside localhost.
