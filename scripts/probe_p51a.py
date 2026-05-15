"""P5.1a verification probe — 2D dispatch at 25M points.

Loads p51a_25m_smoke.ipynb. Validates that:
  - 25M points upload single-shot through the existing comm channel
  - The density build compute pass dispatches in 2D (25M/256 = 97,657 > 65,535)
    without triggering 'exceeds max compute workgroups per dimension'
  - FPS holds > 50

Critical: 1D dispatch at 25M would fail silently (bin_count never populated,
canvas all-background). This probe is the regression gate for the
num_workgroups-based flat-index addressing.
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
TOKEN = "stipple-p51a-token"
NOTEBOOK = "examples/p51a_25m_smoke.ipynb"


def find_free_port(start: int = 9070) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9070")


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


def avg_region_rgb(img_path: Path, cx_frac: float, cy_frac: float, half_px: int = 24):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    cx = int(w * cx_frac)
    cy = int(h * cy_frac)
    region = img.crop((cx - half_px, cy - half_px, cx + half_px, cy + half_px))
    px = list(region.getdata())
    n = max(1, len(px))
    rs = sum(p[0] for p in px) / n
    gs = sum(p[1] for p in px) / n
    bs = sum(p[2] for p in px) / n
    return rs, gs, bs


def run_probe(headed: bool = False) -> dict:
    port = find_free_port()
    print(f"[probe] starting jupyter on :{port}")
    jupyter = start_jupyter(port)
    result: dict = {"port": port}
    dispatch_errors: list[str] = []

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

            def on_console(m):
                txt = m.text
                if "exceeds max compute workgroups" in txt or "Invalid CommandBuffer" in txt:
                    dispatch_errors.append(txt)
                if m.type in ("error", "warning"):
                    print(f"[console] {m.type}: {txt}")

            page.on("console", on_console)

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
                    return outs.some(o => /render_mode\\s*:\\s*density/.test(o.textContent)
                                          && /FPS\\s*:\\s*[\\d.]+/.test(o.textContent));
                }""",
                timeout=240_000,
            )
            time.sleep(2.0)

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            banner = next((o for o in outputs if "render_mode" in o), "")
            result["banner"] = banner[:300]

            m = re.search(r"FPS\s*:\s*([\d.]+)", banner)
            if m:
                result["fps"] = float(m.group(1))
            m = re.search(r"frame_ms\s*:\s*([\d.]+)", banner)
            if m:
                result["frame_ms"] = float(m.group(1))
            m = re.search(r"rows\s*:\s*([\d,]+)", banner)
            if m:
                result["rows"] = int(m.group(1).replace(",", ""))
            m = re.search(r"render_mode\s*:\s*(\w+)", banner)
            if m:
                result["render_mode"] = m.group(1)

            canvas = page.locator(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]'
            ).first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            shot = REPO / "scripts" / "p51a-canvas.png"
            canvas.screenshot(path=str(shot))
            result["screenshot"] = str(shot)
            result["rgb_center"] = tuple(round(v, 1) for v in avg_region_rgb(shot, 0.5, 0.5))
            result["rgb_corner"] = tuple(round(v, 1) for v in avg_region_rgb(shot, 0.05, 0.05))

            result["dispatch_errors"] = dispatch_errors
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
    print("P5.1a PROBE — 2D dispatch at 25M")
    for k in (
        "render_mode", "rows", "fps", "frame_ms",
        "rgb_center", "rgb_corner",
        "dispatch_errors", "screenshot",
    ):
        if k in r:
            print(f"  {k:<16}: {r[k]}")
    print("=" * 60)

    mode_ok = r.get("render_mode") == "density"
    rows_ok = r.get("rows") == 25_000_000
    fps_ok = (r.get("fps") or 0) > 50.0
    center_bright = sum(r.get("rgb_center") or (0, 0, 0)) > 60.0
    corner_dark = sum(r.get("rgb_corner") or (255, 255, 255)) < 90.0
    no_dispatch_err = len(r.get("dispatch_errors") or []) == 0

    print(f"  mode == density: {mode_ok}")
    print(f"  rows == 25M: {rows_ok}")
    print(f"  FPS > 50: {fps_ok}")
    print(f"  center has density pixels: {center_bright}")
    print(f"  corner is empty: {corner_dark}")
    print(f"  no dispatch-limit errors in console: {no_dispatch_err}")

    ok = mode_ok and rows_ok and fps_ok and center_bright and corner_dark and no_dispatch_err
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
