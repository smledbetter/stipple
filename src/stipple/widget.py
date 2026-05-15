from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anywidget
import numpy as np
import pyarrow as pa
import traitlets

from . import _palettes

_STATIC = Path(__file__).parent / "_static"


class Stipple(anywidget.AnyWidget):
    """WebGPU scatter widget.

    Examples
    --------
    Bare arrays:

        w = Stipple(x=xs, y=ys)

    DataFrame columns by name (pandas / polars / pyarrow.Table / dict):

        w = Stipple(df, x="col_x", y="col_y", color="col_c")

    Categorical color (default tab10):

        w = Stipple(x=xs, y=ys, color=labels)
        w.color_categories  # category index -> label

    Continuous color (default viridis):

        w = Stipple(x=xs, y=ys, color=losses, color_kind="continuous")
        w.color_range  # [vmin, vmax] used for the colormap

    Custom palette:

        # Built-in named (matplotlib colormap name if matplotlib installed,
        # otherwise one of: viridis, plasma, magma, inferno, tab10)
        w = Stipple(x=xs, y=ys, color=losses, color_kind="continuous",
                    color_palette="plasma")
        # Or pass an (N, 3) / (N, 4) numpy array in [0, 1]:
        my_palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        w = Stipple(x=xs, y=ys, color=labels, color_palette=my_palette)

    Lasso selection (shift+drag in the canvas):

        def on_change(_):
            print(f"selected {len(w.selected_indices):,} rows")
        w.observe(on_change, names="selection_count")
    """

    _esm = _STATIC / "index.js"

    status = traitlets.Unicode("checking").tag(sync=True)
    adapter_info = traitlets.Unicode("").tag(sync=True)
    error = traitlets.Unicode("").tag(sync=True)
    n_points = traitlets.Int(0).tag(sync=True)

    client_ready = traitlets.Bool(False).tag(sync=True)
    rows_received = traitlets.Int(0).tag(sync=True)
    bytes_received = traitlets.Int(0).tag(sync=True)
    avg_frame_ms = traitlets.Float(0.0).tag(sync=True)
    last_fps = traitlets.Float(0.0).tag(sync=True)

    # Categorical: list of original category labels in code-index order.
    # Continuous: empty (use color_range instead).
    color_categories = traitlets.List(default_value=[]).tag(sync=True)
    # Continuous: [vmin, vmax] used to normalize before quantization.
    # Categorical: empty.
    color_range = traitlets.List(default_value=[]).tag(sync=True)

    selection_count = traitlets.Int(0).tag(sync=True)
    selection_ms = traitlets.Float(0.0).tag(sync=True)
    # When a lasso selection is active, unselected points render at this
    # fraction of their full alpha (0.0–1.0). Default 0.4 = ~halfway dim.
    # Bump toward 1.0 to disable the dim effect entirely.
    selection_dim = traitlets.Float(0.4).tag(sync=True)

    # Render mode:
    #   "scatter"      — one anti-aliased disk per point (great up to ~10M).
    #   "density"      — 1024² bin grid heatmap, log(count) → palette. Pan/
    #                    zoom/lasso still work; hover + selection-dim off.
    #   "density-only" — same render, but drops the positions buffer after
    #                    the bin grid is built. No lasso, no hover. Required
    #                    for very-large-N (~100M+) loads where keeping the
    #                    positions in GPU memory isn't feasible.
    render_mode = traitlets.Enum(
        ["scatter", "density", "density-only"], default_value="scatter"
    ).tag(sync=True)

    def __init__(
        self,
        data: Any = None,
        *,
        x: Any = None,
        y: Any = None,
        color: Any = None,
        color_kind: Literal["categorical", "continuous"] | None = None,
        color_palette: str | np.ndarray | Any = "auto",
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pending: dict[str, Any] | None = None
        self._gen_counter: int = 0
        self._selected_indices: np.ndarray = np.empty(0, dtype=np.uint32)
        self.on_msg(self._on_custom_msg)
        if data is not None:
            x_arr, y_arr, color_arr = _extract_columns(data, x, y, color)
            self._stage(x_arr, y_arr, color_arr, color_kind, color_palette, vmin, vmax)
        elif x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("Stipple needs both x and y, or neither.")
            self._stage(x, y, color, color_kind, color_palette, vmin, vmax)

    @property
    def selected_indices(self) -> np.ndarray:
        """Row indices most recently lassoed in the widget (uint32)."""
        return self._selected_indices

    def _stage(
        self,
        x: Any,
        y: Any,
        color: Any,
        color_kind: str | None,
        color_palette: Any,
        vmin: float | None,
        vmax: float | None,
    ) -> None:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(f"x and y must be 1-D; got {x_arr.shape} and {y_arr.shape}.")
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"x and y length mismatch: {x_arr.shape[0]} vs {y_arr.shape[0]}.")

        n = int(x_arr.shape[0])
        columns: dict[str, np.ndarray] = {"x": x_arr, "y": y_arr}
        palette_rgba: np.ndarray  # (K, 4) float32

        if color is None:
            codes = np.zeros(n, dtype=np.uint32)
            # Density modes sample the palette via log(count) → t ∈ [0, 1],
            # so a single-color palette would map every bin to the same shade.
            # Default to viridis (or the user-specified continuous palette).
            if self.render_mode in ("density", "density-only"):
                pal_arg = color_palette
                if isinstance(pal_arg, str) and pal_arg == "auto":
                    pal_arg = "viridis"
                palette_rgba = _palettes.resolve_palette(pal_arg, want_continuous=True)
            else:
                palette_rgba = np.array(
                    [[0.20, 0.55, 0.95, 1.0]], dtype=np.float32
                )
            columns["color"] = codes
            self.color_categories = []
            self.color_range = []
        else:
            color_arr = np.asarray(color)
            if color_arr.ndim != 1 or color_arr.shape[0] != n:
                raise ValueError(
                    f"color must be 1-D with same length as x/y; got {color_arr.shape}."
                )

            kind = color_kind if color_kind in ("categorical", "continuous") else _infer_color_kind(color_arr)
            if kind not in ("categorical", "continuous"):
                raise ValueError(f"color_kind must be 'categorical' or 'continuous'; got {color_kind!r}.")

            pal_arg = color_palette
            if isinstance(pal_arg, str) and pal_arg == "auto":
                pal_arg = "viridis" if kind == "continuous" else "tab10"
            palette_rgba = _palettes.resolve_palette(pal_arg, want_continuous=(kind == "continuous"))

            if kind == "continuous":
                v = color_arr.astype(np.float32)
                finite = v[np.isfinite(v)]
                lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
                hi = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)
                span = max(hi - lo, 1e-12)
                normalized = np.clip((v - lo) / span, 0.0, 1.0)
                palette_n = int(palette_rgba.shape[0])
                codes = np.clip(
                    np.rint(normalized * (palette_n - 1)), 0, palette_n - 1
                ).astype(np.uint32)
                self.color_categories = []
                self.color_range = [lo, hi]
            else:  # categorical
                codes, cats = _factorize_to_codes(color_arr)
                self.color_categories = [_to_jsonable(c) for c in cats]
                self.color_range = []

            columns["color"] = codes

        # Stash the staged arrays — _push decides single-shot vs chunked at
        # transport time. Computing the world bbox here means the chunked
        # path doesn't need to read positions back on the JS side.
        xmin = float(x_arr.min()) if n else 0.0
        xmax = float(x_arr.max()) if n else 1.0
        ymin = float(y_arr.min()) if n else 0.0
        ymax = float(y_arr.max()) if n else 1.0
        self._pending = {
            "x": np.ascontiguousarray(x_arr, dtype=np.float32),
            "y": np.ascontiguousarray(y_arr, dtype=np.float32),
            "codes": np.ascontiguousarray(columns["color"], dtype=np.uint32),
            "palette": np.ascontiguousarray(palette_rgba, dtype=np.float32).tobytes(),
            "bbox": (xmin, xmax, ymin, ymax),
            "n": n,
        }
        self.n_points = n

        if self.client_ready:
            self._push()

    @traitlets.observe("client_ready")
    def _on_client_ready(self, change: dict[str, Any]) -> None:
        if change["new"] and self._pending is not None:
            self._push()

    # Above this row count, the data is streamed in Arrow IPC chunks via the
    # data_start/data_chunk/data_finalize protocol. Below it, the original
    # single-shot path keeps working for small data.
    _CHUNK_THRESHOLD: int = 4_000_000

    def _push(self) -> None:
        p = self._pending
        if p is None:
            return
        self._pending = None
        n: int = int(p["n"])
        x: np.ndarray = p["x"]
        y: np.ndarray = p["y"]
        codes: np.ndarray = p["codes"]
        palette: bytes = p["palette"]

        if n <= self._CHUNK_THRESHOLD:
            # Single-shot legacy path.
            table = pa.table({"x": x, "y": y, "color": codes})
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            ipc = sink.getvalue().to_pybytes()
            self.send({"type": "data"}, buffers=[ipc, palette])
            return

        # Chunked path. Send a data_start header with bbox + palette, then
        # one Arrow IPC stream per chunk, then a data_finalize sentinel.
        self._gen_counter += 1
        gen = self._gen_counter
        chunk_n = self._CHUNK_THRESHOLD
        n_chunks = (n + chunk_n - 1) // chunk_n

        self.send(
            {
                "type": "data_start",
                "gen": gen,
                "n": n,
                "n_chunks": n_chunks,
                "chunk_n": chunk_n,
                "bbox": list(p["bbox"]),
                "render_mode": self.render_mode,
            },
            buffers=[palette],
        )
        for i in range(n_chunks):
            a = i * chunk_n
            b = min(a + chunk_n, n)
            chunk_table = pa.table(
                {"x": x[a:b], "y": y[a:b], "color": codes[a:b]}
            )
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, chunk_table.schema) as writer:
                writer.write_table(chunk_table)
            chunk_ipc = sink.getvalue().to_pybytes()
            self.send(
                {"type": "data_chunk", "gen": gen, "i": i},
                buffers=[chunk_ipc],
            )
        self.send({"type": "data_finalize", "gen": gen})

    def _on_custom_msg(self, _widget: Any, content: dict[str, Any], buffers: list) -> None:
        if content.get("type") != "selection":
            return
        if not buffers:
            self._selected_indices = np.empty(0, dtype=np.uint32)
        else:
            buf = buffers[0]
            mv = memoryview(buf).tobytes() if not isinstance(buf, (bytes, bytearray)) else buf
            arr = np.frombuffer(mv, dtype=np.uint32).copy()
            self._selected_indices = arr
        self.selection_count = self.selection_count + 1
        self.selection_ms = float(content.get("ms") or 0.0)


def _extract_columns(
    data: Any, x: Any, y: Any, color: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resolve column-name args against a DataFrame-like object.

    Accepts pandas DataFrames, polars DataFrames, pyarrow Tables, and plain
    dicts of array-likes. x/y must be string column names; color may be a
    column name or None.
    """
    if not isinstance(x, str) or not isinstance(y, str):
        raise ValueError(
            "When passing a DataFrame as the first argument, x and y must "
            "be column-name strings, e.g. Stipple(df, x='col_a', y='col_b')."
        )
    if color is not None and not isinstance(color, str):
        raise ValueError(
            "When passing a DataFrame, color must be a column-name string "
            "(or None to skip)."
        )
    names = [x, y] + ([color] if color is not None else [])
    cols = _columns_to_numpy(data, names)
    return cols[x], cols[y], cols[color] if color is not None else None


def _columns_to_numpy(data: Any, names: list[str]) -> dict[str, np.ndarray]:
    if isinstance(data, pa.Table):
        return {n: data.column(n).to_numpy(zero_copy_only=False) for n in names}
    if hasattr(data, "__arrow_c_stream__"):
        # Arrow PyCapsule (pandas >= 2.2, polars >= 0.20, ...). pa.table picks
        # this up automatically and avoids per-column object conversions.
        tbl = pa.table(data)
        return {n: tbl.column(n).to_numpy(zero_copy_only=False) for n in names}
    if hasattr(data, "to_arrow"):
        # Older polars releases predate the PyCapsule interface.
        tbl = data.to_arrow()
        return {n: tbl.column(n).to_numpy(zero_copy_only=False) for n in names}
    if isinstance(data, dict):
        return {n: np.asarray(data[n]) for n in names}
    if hasattr(data, "columns") and hasattr(data, "__getitem__"):
        # Last-resort pandas-style duck typing.
        return {n: np.asarray(data[n]) for n in names}
    raise TypeError(
        f"Stipple doesn't recognize this data type: {type(data).__name__}. "
        "Pass a pandas/polars DataFrame, pyarrow.Table, or dict of arrays."
    )


def _infer_color_kind(arr: np.ndarray) -> str:
    """Categorical for bool/int dtypes, continuous for float."""
    if arr.dtype.kind in ("f",):
        return "continuous"
    return "categorical"


def _factorize_to_codes(arr: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    kind = arr.dtype.kind
    if kind == "b":
        codes = arr.astype(np.uint32)
        return codes, [False, True]
    cats, inv = np.unique(arr, return_inverse=True)
    codes = inv.astype(np.uint32)
    return codes, cats.tolist()


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v
