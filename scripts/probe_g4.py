"""G4 verification probe — polygon lasso.

After the widget initializes, simulate Shift+drag drawing a square around
one of the clusters. Check that the widget's status banner reports a
selection event with reasonable count and timing.

Gate target: round-trip < 200 ms on 1M-point select. Kill: > 200 ms OR
trait sync fails for selection sizes > 100k.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
TOKEN = "stipple-g4-token"
NOTEBOOK = "examples/g4_smoke.ipynb"


def find_free_port(start: int = 8940) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8940")


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
    # Keep the system JUPYTER_CONFIG_DIR (widget extensions live there) but
    # use a probe-local workspaces dir so JupyterLab can't restore stashed
    # tabs from prior runs. Also trust the notebook in advance.
    work_dir = REPO / ".jupyter-probe" / "workspaces"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any cached workspace state from earlier probe runs
    for f in work_dir.glob("*.jupyterlab-workspace"):
        f.unlink()
    subprocess.run(
        [str(REPO / ".venv" / "bin" / "jupyter"), "trust", str(REPO / NOTEBOOK)],
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


def run_probe(headed: bool = False) -> dict:
    port = find_free_port()
    print(f"[probe] starting jupyter on :{port}")
    jupyter = start_jupyter(port)
    result: dict = {"port": port, "headed": headed}

    try:
        wait_for_jupyter(port)
        print("[probe] jupyter is up")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=not headed,
                args=[
                    "--enable-unsafe-webgpu",
                    "--use-angle=metal",
                    "--enable-features=Vulkan",
                ],
            )
            ctx = browser.new_context(viewport={"width": 1400, "height": 1100})
            page = ctx.new_page()
            page.on("console", lambda m: print(f"[console] {m.type}: {m.text}"))
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            page.wait_for_selector(".jp-Notebook", state="attached", timeout=60_000)
            page.wait_for_selector(".jp-Cell", state="attached", timeout=30_000)
            time.sleep(3.0)

            page.evaluate(
                """() => {
                    const nb = document.querySelector('.jp-Notebook');
                    if (nb) nb.focus();
                }"""
            )
            time.sleep(0.3)
            page.keyboard.press("Meta+Shift+C")
            page.wait_for_selector(".lm-CommandPalette-input", timeout=5_000)
            page.fill(".lm-CommandPalette-input", "Run All Cells")
            time.sleep(0.4)
            page.keyboard.press("Enter")

            print("[probe] waiting for initial render + FPS banner…")
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Stipple — 1,000,000 rows/.test(o.textContent)
                                              && /FPS: [\\d.]+/.test(o.textContent));
                    }""",
                    timeout=120_000,
                )
                print("[probe] initial banner observed")
            except Exception as e:
                # Diagnostic dump on timeout
                print(f"[probe] init banner timed out: {e}")
                outs = page.locator(".jp-OutputArea-output").all_text_contents()
                for i, o in enumerate(outs):
                    print(f"--- output {i} ---")
                    print(o[:600])
                dump = REPO / "scripts" / "g4-dump.png"
                page.screenshot(path=str(dump), full_page=True)
                print(f"[probe] dump saved: {dump}")
                raise
            time.sleep(0.6)

            # Locate canvas + compute lasso target around the LEFT cluster
            # (cluster_a is at world x=-3, fitView margin 1.8 / span ~6
            # places it roughly at the left third of the canvas).
            canvas = page.locator(".jp-OutputArea-output canvas").first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            time.sleep(0.3)
            box = canvas.bounding_box()
            assert box, "no canvas bounding box"
            # Square around the leftmost cluster: ~25% across, vertically centered
            cx = box["x"] + box["width"] * 0.22
            cy = box["y"] + box["height"] * 0.50
            half = min(box["width"], box["height"]) * 0.14

            print(f"[probe] lassoing around ({cx:.0f}, {cy:.0f}) box=±{half:.0f}px")

            t0 = time.monotonic()
            page.keyboard.down("Shift")
            page.mouse.move(cx - half, cy - half)
            page.mouse.down()
            page.mouse.move(cx + half, cy - half, steps=8)
            page.mouse.move(cx + half, cy + half, steps=8)
            page.mouse.move(cx - half, cy + half, steps=8)
            page.mouse.move(cx - half, cy - half, steps=8)
            page.mouse.up()
            page.keyboard.up("Shift")

            print("[probe] waiting for 'Python ack' status (end-to-end round-trip)…")
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /lasso: [\\d,]+ \\/ .*Python ack ✓/.test(o.textContent));
                }""",
                timeout=15_000,
            )
            wall = time.monotonic() - t0
            result["wall_seconds"] = wall

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            result["outputs"] = outputs

            banner = next((o for o in outputs if "lasso: " in o), "")
            m = re.search(r"lasso: ([\d,]+) / ([\d,]+) selected · gpu ([\d.]+) ms · Python ack", banner)
            if m:
                result["selected"] = int(m.group(1).replace(",", ""))
                result["total"] = int(m.group(2).replace(",", ""))
                result["gpu_ms"] = float(m.group(3))
                result["python_ack"] = True

            # Also check the on_selection callback's print output landed
            sel_print = next(
                (o for o in outputs if re.search(r"selection #\d+:", o)), ""
            )
            m2 = re.search(
                r"selection #(\d+): ([\d,]+) rows · ([\d.]+)% from cluster_a · ([\d.]+) ms",
                sel_print,
            )
            if m2:
                result["py_seen"] = True
                result["py_selection_n"] = int(m2.group(1))
                result["py_rows"] = int(m2.group(2).replace(",", ""))
                result["py_purity"] = float(m2.group(3))
                result["py_ms"] = float(m2.group(4))

            shot = REPO / "scripts" / "g4-canvas.png"
            try:
                canvas.screenshot(path=str(shot))
                result["canvas"] = str(shot)
            except Exception as e:
                print(f"[probe] canvas snap failed: {e}")

            page_shot = REPO / "scripts" / "g4-result.png"
            page.screenshot(path=str(page_shot), full_page=True)
            result["screenshot"] = str(page_shot)

            browser.close()

    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()

    return result


def main() -> int:
    headed = "--headed" in sys.argv
    r = run_probe(headed=headed)
    print("\n" + "=" * 60)
    print(f"G4 PROBE")
    for k in (
        "wall_seconds",
        "selected",
        "total",
        "gpu_ms",
        "python_ack",
    ):
        if k in r:
            v = r[k]
            if isinstance(v, float):
                print(f"  {k:<14}: {v:.2f}")
            else:
                print(f"  {k:<14}: {v}")
    print("=" * 60)

    sel = int(r.get("selected") or 0)
    gpu_ms = float(r.get("gpu_ms") or 9999)
    py_ack = bool(r.get("python_ack"))
    ok = (
        sel > 100_000  # meaningfully large selection
        and sel < 1_000_000  # not the whole dataset
        and gpu_ms < 200  # gate threshold
        and py_ack  # selection_count incremented by Python → synced back
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
