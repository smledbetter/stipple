"""Record the 10M FineWeb 'paint by lasso' demo as a real-time MP4.

Workflow:
  1. Headless Chromium opens `examples/demo_fineweb_paint.ipynb`
  2. Run All Cells (10M points load → scatter view in uniform purple)
  3. Shift+drag a lasso around a cluster
  4. Re-run the paint cell — `w.update_color(sims)` recolors all 10M
     points by Gaussian closeness to the lasso centroid in a single
     GPU buffer swap

Uses Playwright's WebM recorder. The recolor is one layout-affecting
event so the Chromium compositor catches it reliably without the
screenshot-loop slowdown we needed for in-flight zoom motion.
"""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent
TOKEN = "stipple-rec-token"
NOTEBOOK = "examples/demo_fineweb_paint.ipynb"
REC_DIR = REPO / "scripts" / "recordings" / datetime.now().strftime("%Y%m%d-%H%M%S")
VIEWPORT = {"width": 1280, "height": 860}


CURSOR_CSS_JS = """() => {
  if (document.getElementById('rec-cursor')) return;
  const cur = document.createElement('div');
  cur.id = 'rec-cursor';
  cur.style.cssText = `
    position: fixed; z-index: 99999; pointer-events: none;
    width: 22px; height: 22px;
    border: 3px solid #ff3b30; border-radius: 50%;
    background: rgba(255, 59, 48, 0.4);
    box-shadow: 0 0 14px rgba(255, 59, 48, 0.7);
    transform: translate(-50%, -50%);
    left: -100px; top: -100px;
  `;
  document.body.appendChild(cur);
  const move = (e) => {
    cur.style.left = e.clientX + 'px';
    cur.style.top = e.clientY + 'px';
  };
  window.addEventListener('mousemove', move, true);
  window.addEventListener('pointermove', move, true);
  const down = () => {
    cur.style.background = 'rgba(255, 59, 48, 0.85)';
    cur.style.transform = 'translate(-50%, -50%) scale(0.7)';
  };
  const up = () => {
    cur.style.background = 'rgba(255, 59, 48, 0.4)';
    cur.style.transform = 'translate(-50%, -50%) scale(1.0)';
  };
  window.addEventListener('mousedown', down, true);
  window.addEventListener('pointerdown', down, true);
  window.addEventListener('mouseup', up, true);
  window.addEventListener('pointerup', up, true);
}"""


def find_free_port(start: int = 9150) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9150")


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


def record() -> Path:
    REC_DIR.mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    print(f"[rec] starting jupyter on :{port}")
    print(f"[rec] video → {REC_DIR}")
    jupyter = start_jupyter(port)

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
            ctx = browser.new_context(
                viewport=VIEWPORT,
                record_video_dir=str(REC_DIR),
                record_video_size=VIEWPORT,
            )
            page = ctx.new_page()
            page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))

            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector(".jp-Notebook", state="attached", timeout=60_000)
            page.wait_for_selector(".jp-Cell", state="attached", timeout=30_000)
            time.sleep(1.5)

            page.evaluate(CURSOR_CSS_JS)
            time.sleep(0.2)

            def palette(query: str) -> None:
                page.evaluate("() => document.querySelector('.jp-Notebook')?.focus()")
                time.sleep(0.2)
                for _ in range(3):
                    page.keyboard.press("Meta+Shift+C")
                    try:
                        page.wait_for_selector(
                            ".lm-CommandPalette-input", state="visible", timeout=2_000
                        )
                        break
                    except Exception:
                        time.sleep(0.3)
                page.fill(".lm-CommandPalette-input", query)
                time.sleep(0.3)
                page.keyboard.press("Enter")
                time.sleep(0.4)

            print("[rec] running all cells…")
            palette("Run All Cells")

            print("[rec] waiting for widget canvas to mount…")
            page.wait_for_selector(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]',
                state="attached",
                timeout=120_000,
            )
            canvas = page.locator(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]'
            ).first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            time.sleep(0.5)

            print("[rec] waiting for 10M scatter to settle…")
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /FPS:\\s*[\\d.]+/.test(o.textContent)
                                          && /mode=scatter/.test(o.textContent));
                }""",
                timeout=300_000,
            )
            # Brief beat so the viewer can register the loaded scatter.
            time.sleep(1.5)

            def center_canvas() -> None:
                page.evaluate("""() => {
                    const c = document.querySelector('canvas[data-stipple-role=\"render\"]');
                    if (c) c.scrollIntoView({ block: 'center', behavior: 'instant' });
                }""")

            center_canvas()
            time.sleep(0.4)
            box = canvas.bounding_box()
            assert box, "no canvas bounding box"
            # Lasso a region biased to one side so the radial gradient
            # has visible falloff in every direction after recolor.
            cx = box["x"] + box["width"] * 0.42
            cy = box["y"] + box["height"] * 0.48
            radius = min(box["width"], box["height"]) * 0.10

            print("[rec] lassoing a cluster…")
            page.keyboard.down("Shift")
            start_x, start_y = cx + radius, cy
            page.mouse.move(start_x, start_y)
            time.sleep(0.25)
            page.mouse.down()
            n_steps = 40
            for i in range(1, n_steps + 1):
                ang = (i / n_steps) * 2 * math.pi
                page.mouse.move(
                    cx + radius * math.cos(ang),
                    cy + radius * math.sin(ang),
                    steps=2,
                )
                time.sleep(0.060)
            page.mouse.move(start_x, start_y, steps=4)
            page.mouse.up()
            page.keyboard.up("Shift")

            print("[rec] waiting for Python ack…")
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /Python ack ✓/.test(o.textContent));
                }""",
                timeout=30_000,
            )
            # Brief beat after lasso commit — viewer sees the static
            # purple scatter + lasso outline before the recolor lands.
            time.sleep(1.2)

            print("[rec] re-running paint cell…")
            paint_cell = page.locator(
                ".jp-CodeCell", has_text="w.update_color(sims)"
            ).first
            paint_cell.scroll_into_view_if_needed(timeout=5_000)
            time.sleep(0.6)
            paint_cell.locator(".cm-editor").first.click()
            time.sleep(0.4)
            page.keyboard.press("Escape")
            time.sleep(0.3)
            page.keyboard.press("Shift+Enter")

            print("[rec] waiting for recolor to complete…")
            page.wait_for_function(
                """() => {
                    const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                    const t = cells.find(c =>
                        (c.querySelector('.cm-content')?.textContent || '')
                            .includes('w.update_color(sims)')
                    );
                    if (!t) return false;
                    const out = t.querySelector('.jp-OutputArea-output');
                    return out && /repainted in one GPU upload/.test(out.textContent);
                }""",
                timeout=60_000,
            )
            # Final beat — viewer absorbs the Gaussian heatmap result.
            center_canvas()
            time.sleep(5.0)

            print("[rec] done, closing context to flush video…")
            ctx.close()
            browser.close()

    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()

    webms = sorted(REC_DIR.glob("*.webm"))
    if not webms:
        raise RuntimeError("no WebM produced — playwright recording failed")
    webm = webms[0]
    mp4 = REC_DIR / "fineweb-paint-demo.mp4"
    print(f"[rec] transcoding WebM → MP4…")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(webm),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    return mp4


def main() -> int:
    out = record()
    sz = out.stat().st_size / (1024 * 1024)
    print(f"\nrecording: {out}  ({sz:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
