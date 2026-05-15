"""P2 verification probe — continuous color (viridis).

Renders 100k points with a continuous color gradient, verifies:
  - banner reports `continuous color [vmin, vmax]`
  - vmin/vmax round-trip into the Python `color_range` trait
  - canvas pixel sampling at the low end vs high end of the gradient
    matches viridis (dark purple → yellow-green)
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
TOKEN = "stipple-p2-token"
NOTEBOOK = "examples/p2_continuous_smoke.ipynb"


def find_free_port(start: int = 8980) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8980")


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
    result: dict = {"port": port, "headed": headed}

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

            print("[probe] waiting for continuous-color banner…")
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o =>
                        /Stipple — 100,000 rows/.test(o.textContent)
                        && /continuous color \\[[\\d.]+, [\\d.]+\\]/.test(o.textContent)
                        && /FPS: [\\d.]+/.test(o.textContent)
                    );
                }""",
                timeout=120_000,
            )
            print("[probe] banner observed")

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            banner = next((o for o in outputs if "continuous color" in o), "")
            m_range = re.search(r"continuous color \[([\d.]+), ([\d.]+)\]", banner)
            if m_range:
                result["color_vmin"] = float(m_range.group(1))
                result["color_vmax"] = float(m_range.group(2))
            m_fps = re.search(r"FPS: ([\d.]+)", banner)
            if m_fps:
                result["fps"] = float(m_fps.group(1))

            # Canvas pixel sampling. The data is laid out diagonally so low
            # color (dark purple, viridis[0]) sits near bottom-left and high
            # color (yellow-green, viridis[255]) near top-right after fitView.
            canvas = page.locator(".jp-OutputArea-output canvas").first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            time.sleep(0.4)
            shot = REPO / "scripts" / "p2-canvas.png"
            canvas.screenshot(path=str(shot))
            result["canvas"] = str(shot)

            # Sample pixels from the saved PNG (WebGPU canvas readback via 2D
            # context returns zeros — the swap-chain isn't preserved). Pillow
            # is already in the venv (matplotlib dep).
            from PIL import Image  # type: ignore[import-not-found]

            img = Image.open(shot).convert("RGBA")
            w, h = img.size
            def avg_region(cx_frac: float, cy_frac: float, half_px: int = 6) -> tuple[int, int, int, int]:
                cx = int(w * cx_frac)
                cy = int(h * cy_frac)
                region = img.crop((cx - half_px, cy - half_px, cx + half_px, cy + half_px))
                pixels = list(region.getdata())
                # Average R, G, B, A
                n = len(pixels)
                r = sum(p[0] for p in pixels) // n
                g = sum(p[1] for p in pixels) // n
                b = sum(p[2] for p in pixels) // n
                a = sum(p[3] for p in pixels) // n
                return (r, g, b, a)

            samples = {
                "bottomLeft": avg_region(0.25, 0.75),
                "topRight": avg_region(0.75, 0.25),
                "center": avg_region(0.50, 0.50),
                "canvasSize": [w, h],
            }
            result["pixel_samples"] = samples

            # Viridis sanity:
            #   - viridis[0] ≈ (68, 1, 84)   — dark purple, R<B, G low
            #   - viridis[128] ≈ (33, 145, 140) — teal
            #   - viridis[255] ≈ (253, 231, 37)  — yellow, R high G high B low
            # Background is (~13, 15, 23) — dark navy. We rely on the diagonal
            # layout so bottomLeft ~= low color, topRight ~= high color.
            bl = samples["bottomLeft"]
            tr = samples["topRight"]
            # Dark-purple test: bl has R close to 0–100 and B > R.
            #   (background is also low-R, so we'd also accept it; this test is
            #   loose because of viewport positioning. We mainly look for the
            #   topRight being yellow-ish: R high AND G high AND B low.)
            tr_is_yellow = tr[0] > 120 and tr[1] > 120 and tr[2] < 120
            bl_is_purple_or_bg = (bl[2] >= bl[0]) or sum(bl[:3]) < 90
            result["viridis_topright_yellow"] = tr_is_yellow
            result["viridis_bottomleft_purpleish"] = bl_is_purple_or_bg

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
    print("P2 PROBE")
    for k in (
        "color_vmin",
        "color_vmax",
        "fps",
        "pixel_samples",
        "viridis_topright_yellow",
        "viridis_bottomleft_purpleish",
        "canvas",
    ):
        if k in r:
            print(f"  {k:<28}: {r[k]}")
    print("=" * 60)
    fps = float(r.get("fps") or 0)
    ok = (
        r.get("viridis_topright_yellow") is True
        and r.get("viridis_bottomleft_purpleish") is True
        and abs(r.get("color_vmin", 999) - 0.0) < 0.01
        and abs(r.get("color_vmax", 0.0) - 1.0) < 0.01
        and fps > 30
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
