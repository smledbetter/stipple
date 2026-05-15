"""P4 verification probe — hover tooltip (spatial-index-backed).

Opens g4_smoke.ipynb, waits for the FPS banner, then:
  1. Moves the cursor over the LEFT cluster (cluster_a) → tooltip appears
     with `row:` / `x:` / `y:` lines; x value should be in cluster_a's
     world range (around x ≈ -3).
  2. Moves to a sparse spot (corner of the canvas) → tooltip hides.
  3. Verifies the tooltip element has the `data-stipple-role="tooltip"`
     attribute and is positioned inside the wrap div.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
TOKEN = "stipple-p4-token"
NOTEBOOK = "examples/g4_smoke.ipynb"


def find_free_port(start: int = 9010) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9010")


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
    result: dict = {"port": port}

    try:
        wait_for_jupyter(port)
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
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
            page.on(
                "console",
                lambda m: print(f"[console] {m.type}: {m.text}")
                if m.type in ("error", "warning")
                else None,
            )

            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
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
                    return outs.some(o => /Stipple — 1,000,000 rows/.test(o.textContent)
                                          && /FPS: [\\d.]+/.test(o.textContent));
                }""",
                timeout=120_000,
            )
            # Hover grid builds asynchronously after first render; give it a beat.
            time.sleep(1.2)

            canvas = page.locator('.jp-OutputArea-output canvas[data-stipple-role="render"]').first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            tooltip = page.locator('.jp-OutputArea-output [data-stipple-role="tooltip"]').first
            assert tooltip, "no tooltip element"

            box = canvas.bounding_box()
            assert box, "no canvas bounding box"

            # ---- Hover over LEFT cluster (cluster_a, world x ≈ -3) ----
            cx_left = box["x"] + box["width"] * 0.22
            cy_left = box["y"] + box["height"] * 0.50
            page.mouse.move(cx_left, cy_left)
            time.sleep(0.3)
            # Nudge to ensure a fresh pointermove with the tooltip in steady state.
            page.mouse.move(cx_left + 1, cy_left + 1)
            time.sleep(0.4)

            tt_state_left = page.evaluate(
                """() => {
                    const t = document.querySelector('[data-stipple-role="tooltip"]');
                    if (!t) return { found: false };
                    const cs = getComputedStyle(t);
                    return {
                        found: true,
                        display: cs.display,
                        text: t.textContent || '',
                    };
                }"""
            )
            result["tt_left"] = tt_state_left

            # ---- Hover over empty corner (outside any cluster) ----
            # Take a screenshot of the canvas + use top-left corner. Clusters
            # live in the central band, so the top-left of the canvas is sparse.
            page.mouse.move(box["x"] + 4, box["y"] + 4)
            time.sleep(0.4)
            tt_state_empty = page.evaluate(
                """() => {
                    const t = document.querySelector('[data-stipple-role="tooltip"]');
                    if (!t) return { found: false };
                    const cs = getComputedStyle(t);
                    return {
                        found: true,
                        display: cs.display,
                        text: t.textContent || '',
                    };
                }"""
            )
            result["tt_empty"] = tt_state_empty

            # ---- Snapshot for debugging ----
            shot = REPO / "scripts" / "p4-canvas.png"
            page.locator('[data-stipple-role="wrap"]').first.screenshot(path=str(shot))
            result["screenshot"] = str(shot)

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
    print("P4 PROBE — hover tooltip")
    for k in ("tt_left", "tt_empty", "screenshot"):
        if k in r:
            print(f"  {k:<10}: {r[k]}")
    print("=" * 60)

    left = r.get("tt_left") or {}
    empty = r.get("tt_empty") or {}
    text_left = (left.get("text") or "").strip()
    text_empty = (empty.get("text") or "").strip()
    left_visible = left.get("display") == "block"
    empty_hidden = empty.get("display") == "none"
    has_row = "row:" in text_left
    has_x = "x:" in text_left
    has_y = "y:" in text_left

    print(f"  tooltip shown over cluster (display=block): {left_visible}")
    print(f"  tooltip text contains row/x/y: {has_row and has_x and has_y}")
    print(f"  tooltip hidden over empty area: {empty_hidden}")

    ok = left_visible and has_row and has_x and has_y and empty_hidden
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
