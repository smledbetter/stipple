"""P3 verification probe — selection highlight (lasso dims unselected).

Opens g4_smoke.ipynb (1M points, 3 colored clusters). Lassos around the
LEFT cluster only. Then samples pixels:
  - INSIDE the lassoed region (cluster_a, blue) → should stay bright
  - OUTSIDE the lassoed region (cluster_b, orange / cluster_c, green) →
    should dim per `selection_dim`

Compares both regions to a pre-lasso baseline snapshot to confirm only the
unselected regions dimmed.
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
TOKEN = "stipple-p3-token"
NOTEBOOK = "examples/g4_smoke.ipynb"


def find_free_port(start: int = 8990) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8990")


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


def avg_region_brightness(img_path: Path, cx_frac: float, cy_frac: float, half_px: int = 12) -> float:
    """Mean luminance of an HxH region around a fraction-of-canvas coord."""
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    cx = int(w * cx_frac)
    cy = int(h * cy_frac)
    region = img.crop((cx - half_px, cy - half_px, cx + half_px, cy + half_px))
    px = list(region.getdata())
    n = len(px)
    return sum((r + g + b) / 3 for (r, g, b) in px) / n


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
            page.on("console", lambda m: print(f"[console] {m.type}: {m.text}") if "stipple" in m.text.lower() or m.type in ("error", "warning") else None)

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
            time.sleep(0.6)

            canvas = page.locator('.jp-OutputArea-output canvas[data-stipple-role="render"]').first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            time.sleep(0.4)

            # ---- Baseline (pre-lasso) snapshot ----
            baseline = REPO / "scripts" / "p3-canvas-baseline.png"
            canvas.screenshot(path=str(baseline))
            result["baseline"] = str(baseline)
            box = canvas.bounding_box()
            assert box, "no canvas bounding box"

            # ---- Lasso the LEFT cluster (cluster_a at world x=-3).
            # After fitView, the leftmost cluster sits roughly at canvas x≈0.22.
            cx = box["x"] + box["width"] * 0.22
            cy = box["y"] + box["height"] * 0.50
            half = min(box["width"], box["height"]) * 0.14

            page.keyboard.down("Shift")
            page.mouse.move(cx - half, cy - half)
            page.mouse.down()
            page.mouse.move(cx + half, cy - half, steps=8)
            page.mouse.move(cx + half, cy + half, steps=8)
            page.mouse.move(cx - half, cy + half, steps=8)
            page.mouse.move(cx - half, cy - half, steps=8)
            page.mouse.up()
            page.keyboard.up("Shift")

            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /Python ack ✓/.test(o.textContent));
                }""",
                timeout=15_000,
            )
            time.sleep(0.4)  # let render complete

            after = REPO / "scripts" / "p3-canvas-after.png"
            canvas.screenshot(path=str(after))
            result["after"] = str(after)

            # ---- Compare brightness in the LEFT (selected) vs RIGHT (unselected) regions
            # Coordinates are within-canvas fractions; sample slightly off the
            # exact lasso edges so we measure cluster density, not lasso line.
            left_baseline = avg_region_brightness(baseline, 0.18, 0.55)
            left_after = avg_region_brightness(after, 0.18, 0.55)
            right_baseline = avg_region_brightness(baseline, 0.78, 0.55)
            right_after = avg_region_brightness(after, 0.78, 0.55)
            top_baseline = avg_region_brightness(baseline, 0.5, 0.25)
            top_after = avg_region_brightness(after, 0.5, 0.25)

            result["left_baseline"] = round(left_baseline, 2)
            result["left_after"] = round(left_after, 2)
            result["right_baseline"] = round(right_baseline, 2)
            result["right_after"] = round(right_after, 2)
            result["top_baseline"] = round(top_baseline, 2)
            result["top_after"] = round(top_after, 2)

            # Pass criteria:
            # - Left (selected) brightness should be approximately preserved
            #   (allow up to ~10% drop from anti-aliasing / lasso line bleed)
            # - Right (unselected) brightness should drop meaningfully
            #   (≥ 30% drop with default selection_dim=0.4)
            # - Top (unselected) brightness should drop meaningfully too
            left_ratio = left_after / max(left_baseline, 1e-3)
            right_ratio = right_after / max(right_baseline, 1e-3)
            top_ratio = top_after / max(top_baseline, 1e-3)
            result["left_ratio"] = round(left_ratio, 3)
            result["right_ratio"] = round(right_ratio, 3)
            result["top_ratio"] = round(top_ratio, 3)

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
    print("P3 PROBE — selection-highlight contrast")
    for k in (
        "left_baseline",
        "left_after",
        "left_ratio",
        "right_baseline",
        "right_after",
        "right_ratio",
        "top_baseline",
        "top_after",
        "top_ratio",
        "baseline",
        "after",
    ):
        if k in r:
            print(f"  {k:<16}: {r[k]}")
    print("=" * 60)
    # PASS criteria
    left_ok = (r.get("left_ratio") or 0) > 0.80   # selected stayed bright
    right_ok = (r.get("right_ratio") or 1) < 0.75  # unselected dimmed visibly
    top_ok = (r.get("top_ratio") or 1) < 0.75
    print(f"  selected stayed bright (left_ratio > 0.80): {left_ok}")
    print(f"  unselected right dimmed (right_ratio < 0.75): {right_ok}")
    print(f"  unselected top dimmed   (top_ratio   < 0.75): {top_ok}")
    return 0 if (left_ok and right_ok and top_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
