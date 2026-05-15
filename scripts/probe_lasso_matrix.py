"""Lasso-at-scale variation matrix for FineWeb.

Diagnostic harness that runs the same end-to-end test (load → density
render → lasso → re-run inspect cell → count selected indices) with
different parameters, to localize the failure mode at 10M.

Variables we sweep:
  - N: number of FineWeb rows loaded (controls single-shot vs chunked)
  - lasso_half: canvas-half-width of the lasso square (controls
    selection-indices buffer size)
  - settle: seconds to wait after load before lassoing

Each variation generates its own temp notebook, drives a headless
JupyterLab + WebGPU Chrome, parses the inspect cell, and records
the JS-side selected count vs the Python-observed `selected_indices`
length. Discrepancies localize where the JS→Python lasso path breaks.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
TOKEN = "stipple-matrix-token"
NB_PATH = REPO / "examples" / "_matrix_tmp.ipynb"


@dataclass
class Variation:
    n_rows: int
    lasso_half: float
    settle: float
    tag: str
    # If True, restart the kernel between data load and lasso to clear
    # any leftover JS stream state.
    restart_kernel: bool = False


VARIATIONS = [
    # Baseline: known-good 1M single-shot, large lasso.
    Variation(n_rows=1_000_000, lasso_half=0.10, settle=4.0, tag="1M_baseline"),
    # Smallest chunked: 5M (chunk threshold is 4M).
    Variation(n_rows=5_000_000, lasso_half=0.10, settle=4.0, tag="5M_chunked_large"),
    # Tiny lasso at 10M — isolates "is it buffer size?"
    Variation(n_rows=10_000_000, lasso_half=0.015, settle=4.0, tag="10M_tiny_lasso"),
    # Long settle at 10M — isolates "is it timing?"
    Variation(n_rows=10_000_000, lasso_half=0.10, settle=30.0, tag="10M_long_settle"),
    # Baseline failure repro at 10M for cross-reference.
    Variation(n_rows=10_000_000, lasso_half=0.10, settle=4.0, tag="10M_default"),
]


NOTEBOOK_TEMPLATE = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "load",
            "metadata": {},
            "outputs": [],
            "source": [
                "import mmap\n",
                "import numpy as np\n",
                "from pathlib import Path\n",
                "from stipple import Stipple\n",
                "DATA = Path.home() / 'Sites/Thinking/Projects/Research/Stipple/app/public/data'\n",
                "arr = np.fromfile(DATA / 'fineweb-10m-gpu-chunk00.f32', dtype=np.float32).reshape(-1, 10)[:__N__]\n",
                "xs = np.ascontiguousarray(arr[:, 0])\n",
                "ys = np.ascontiguousarray(arr[:, 1])\n",
                "offsets = np.fromfile(DATA / 'fineweb-10m-gpu-chunk00-text-offsets.bin', dtype=np.uint64)\n",
                "_blob_f = open(DATA / 'fineweb-10m-gpu-chunk00-text-blob.bin', 'rb')\n",
                "text_blob = mmap.mmap(_blob_f.fileno(), 0, access=mmap.ACCESS_READ)\n",
                "print(f'{len(xs):,} FineWeb documents loaded')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "render",
            "metadata": {},
            "outputs": [],
            "source": [
                "w = Stipple(x=xs, y=ys, render_mode='density')\n",
                "w\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "inspect",
            "metadata": {},
            "outputs": [],
            "source": [
                "idx = w.selected_indices\n",
                "print(f'selected_indices_len={len(idx)}')\n",
                "print(f'selection_count={w.selection_count}')\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write_notebook(n_rows: int) -> None:
    nb = json.loads(json.dumps(NOTEBOOK_TEMPLATE))  # deep copy
    src = nb["cells"][0]["source"]
    nb["cells"][0]["source"] = [s.replace("__N__", f"{n_rows:_}") for s in src]
    NB_PATH.write_text(json.dumps(nb, indent=1))


def find_free_port(start: int = 9130) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9130")


def wait_for_jupyter(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/api?token={TOKEN}"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"jupyter at :{port} didn't come up within {timeout}s")


def start_jupyter(port: int) -> subprocess.Popen:
    work_dir = REPO / ".jupyter-probe" / "workspaces"
    work_dir.mkdir(parents=True, exist_ok=True)
    for f in work_dir.glob("*.jupyterlab-workspace"):
        f.unlink()
    subprocess.run(
        [str(REPO / ".venv" / "bin" / "jupyter"), "trust", str(NB_PATH)],
        check=False,
        capture_output=True,
    )
    env = os.environ.copy()
    env["JUPYTER_DISABLE_BROWSER_OPEN"] = "1"
    return subprocess.Popen(
        [
            str(REPO / ".venv" / "bin" / "jupyter"),
            "lab",
            "--no-browser",
            f"--port={port}",
            f"--IdentityProvider.token={TOKEN}",
            "--ServerApp.disable_check_xsrf=True",
            f"--notebook-dir={REPO}",
            "--allow-root",
            f"--LabApp.workspaces_dir={work_dir}",
        ],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def run_one(v: Variation) -> dict:
    write_notebook(v.n_rows)
    port = find_free_port()
    jupyter = start_jupyter(port)
    result: dict = {"tag": v.tag, "n_rows": v.n_rows, "lasso_half": v.lasso_half, "settle": v.settle}

    try:
        wait_for_jupyter(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--enable-unsafe-webgpu",
                    "--use-angle=metal",
                    "--enable-features=Vulkan",
                ],
            )
            ctx = browser.new_context(viewport={"width": 1400, "height": 1100})
            page = ctx.new_page()
            page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))

            nb_url = f"http://127.0.0.1:{port}/lab/tree/examples/_matrix_tmp.ipynb?token={TOKEN}"
            page.goto(nb_url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector(".jp-Notebook", state="attached", timeout=60_000)
            page.wait_for_selector(".jp-Cell", state="attached", timeout=30_000)
            time.sleep(3.0)

            page.evaluate("() => document.querySelector('.jp-Notebook')?.focus()")
            time.sleep(0.3)
            page.keyboard.press("Meta+Shift+C")
            page.wait_for_selector(".lm-CommandPalette-input", timeout=5_000)
            page.fill(".lm-CommandPalette-input", "Run All Cells")
            time.sleep(0.4)
            page.keyboard.press("Enter")

            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /FPS:\\s*[\\d.]+/.test(o.textContent)
                                          && /Stipple\\s+[—-]\\s+[\\d,]+\\s+rows/.test(o.textContent));
                }""",
                timeout=300_000,
            )
            time.sleep(v.settle)

            canvas = page.locator(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]'
            ).first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            box = canvas.bounding_box()
            assert box

            cx = box["x"] + box["width"] * 0.50
            cy = box["y"] + box["height"] * 0.50
            half = min(box["width"], box["height"]) * v.lasso_half

            page.keyboard.down("Shift")
            page.mouse.move(cx - half, cy - half)
            page.mouse.down()
            page.mouse.move(cx + half, cy - half, steps=10)
            page.mouse.move(cx + half, cy + half, steps=10)
            page.mouse.move(cx - half, cy + half, steps=10)
            page.mouse.move(cx - half, cy - half, steps=10)
            page.mouse.up()
            page.keyboard.up("Shift")

            # Wait for JS lasso banner. Then collect JS-side selected count.
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /lasso:\\s*[\\d,]+/.test(o.textContent));
                    }""",
                    timeout=20_000,
                )
                result["js_fired"] = True
            except Exception:
                result["js_fired"] = False
                browser.close()
                return result

            # Parse JS-side selected count.
            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            banner = next((o for o in outputs if "lasso:" in o), "")
            m = re.search(r"lasso:\s*([\d,]+)\s*/\s*([\d,]+)", banner)
            if m:
                result["js_selected"] = int(m.group(1).replace(",", ""))

            # Wait for Python ack (if it ever arrives). Bounded probe so we
            # don't block forever — separately we re-run the inspect cell to
            # see what Python actually has.
            python_acked = False
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Python ack ✓/.test(o.textContent));
                    }""",
                    timeout=15_000,
                )
                python_acked = True
            except Exception:
                pass
            result["python_acked"] = python_acked

            # Re-run inspect cell.
            inspect_cell = page.locator(
                ".jp-CodeCell", has_text="selected_indices_len"
            ).first
            inspect_cell.locator(".cm-editor").first.click()
            time.sleep(0.3)
            page.keyboard.press("Escape")
            time.sleep(0.2)
            page.keyboard.press("Shift+Enter")

            # Read inspect output (up to 30s).
            t0 = time.monotonic()
            inspect_text = ""
            while time.monotonic() - t0 < 30.0:
                inspect_text = page.evaluate(
                    """() => {
                        const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                        const target = cells.find(c =>
                            (c.querySelector('.cm-content')?.textContent || '').includes('selected_indices_len')
                        );
                        if (!target) return '';
                        const out = target.querySelector('.jp-OutputArea-output');
                        return out ? out.textContent : '';
                    }"""
                )
                if "selected_indices_len=" in inspect_text:
                    break
                time.sleep(0.5)
            result["inspect_text"] = inspect_text[:200]
            m = re.search(r"selected_indices_len=(\d+)", inspect_text)
            if m:
                result["py_indices"] = int(m.group(1))
            m = re.search(r"selection_count=(\d+)", inspect_text)
            if m:
                result["py_selection_count"] = int(m.group(1))

            browser.close()
    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()
    return result


def main() -> int:
    selected = sys.argv[1:] or [v.tag for v in VARIATIONS]
    results = []
    for v in VARIATIONS:
        if v.tag not in selected:
            continue
        print(f"\n=== Running variation: {v.tag} (N={v.n_rows:,}, lasso_half={v.lasso_half}, settle={v.settle}s) ===")
        t0 = time.monotonic()
        r = run_one(v)
        r["wall_s"] = round(time.monotonic() - t0, 1)
        results.append(r)
        print(f"  → {r}")

    print("\n" + "=" * 90)
    print(f"{'tag':<22} {'n_rows':>12} {'lasso':>7} {'settle':>7} {'js':>4} {'jsSel':>8} {'pyAck':>6} {'pyIdx':>8} {'pySel#':>7}")
    for r in results:
        print(
            f"{r['tag']:<22} {r['n_rows']:>12,} {r['lasso_half']:>7.3f} {r['settle']:>7.1f} "
            f"{str(r.get('js_fired','?')):>4} {r.get('js_selected','-'):>8} "
            f"{str(r.get('python_acked','?')):>6} {r.get('py_indices','-'):>8} {r.get('py_selection_count','-'):>7}"
        )
    print("=" * 90)
    # Clean up the temp notebook.
    if NB_PATH.exists():
        NB_PATH.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
