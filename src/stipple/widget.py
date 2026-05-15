from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import pyarrow as pa
import traitlets

_STATIC = Path(__file__).parent / "_static"


class Stipple(anywidget.AnyWidget):
    """WebGPU scatter widget.

    Examples
    --------
    Minimal:

        w = Stipple(x=xs, y=ys)

    Colored by a categorical label:

        w = Stipple(x=xs, y=ys, color=labels)
        w.color_categories   # list mapping category index -> label

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

    color_categories = traitlets.List(default_value=[]).tag(sync=True)

    selection_count = traitlets.Int(0).tag(sync=True)
    selection_ms = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        *,
        x: Any = None,
        y: Any = None,
        color: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._pending_ipc: bytes | None = None
        self._selected_indices: np.ndarray = np.empty(0, dtype=np.uint32)
        # NOTE: don't override `_handle_msg` — that name is reserved by
        # ipywidgets.Widget for the comm dispatcher. Register our handler
        # via on_msg so ipywidgets routes only custom messages to us.
        self.on_msg(self._on_custom_msg)
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("Stipple needs both x and y, or neither.")
            self._stage(x, y, color)

    @property
    def selected_indices(self) -> np.ndarray:
        """Row indices most recently lassoed in the widget (uint32)."""
        return self._selected_indices

    def _stage(self, x: Any, y: Any, color: Any) -> None:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(f"x and y must be 1-D; got {x_arr.shape} and {y_arr.shape}.")
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"x and y length mismatch: {x_arr.shape[0]} vs {y_arr.shape[0]}.")

        columns: dict[str, np.ndarray] = {"x": x_arr, "y": y_arr}

        if color is not None:
            color_arr = np.asarray(color)
            if color_arr.ndim != 1 or color_arr.shape[0] != x_arr.shape[0]:
                raise ValueError(
                    f"color must be 1-D with same length as x/y; got {color_arr.shape}."
                )
            codes, cats = _factorize_to_codes(color_arr)
            columns["color"] = codes
            self.color_categories = [_to_jsonable(c) for c in cats]
        else:
            self.color_categories = []

        table = pa.table(columns)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        self._pending_ipc = sink.getvalue().to_pybytes()
        self.n_points = int(x_arr.shape[0])

        if self.client_ready:
            self._push()

    @traitlets.observe("client_ready")
    def _on_client_ready(self, change: dict[str, Any]) -> None:
        if change["new"] and self._pending_ipc is not None:
            self._push()

    def _push(self) -> None:
        if self._pending_ipc is None:
            return
        payload = self._pending_ipc
        self._pending_ipc = None
        self.send({"type": "data"}, buffers=[payload])

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
