"""P5.1b verification probe — chunked Arrow IPC + density-only mode.

Runs against two notebooks (selectable via --notebook):
  - synthetic 100M Gaussian clusters (default)
  - real TPC-H lineitem at SF=25 (149,996,355 rows) — `--notebook tpch`

Verifies:
  - chunked upload completes (final FPS banner appears)
  - render_mode reported as density-only
  - rows_received matches expected N
  - canvas has visible density-rendered pixels (center sample non-empty)
  - FPS > 50
"""

from __future__ import annotations

import argparse
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
TOKEN = "stipple-p51b-token"


def find_free_port(start: int = 9090) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9090")


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


def start_jupyter(port: int, notebook: str) -> subprocess.Popen:
    work_dir = REPO / ".jupyter-probe" / "workspaces"
    work_dir.mkdir(parents=True, exist_ok=True)
    for f in work_dir.glob("*.jupyterlab-workspace"):
        f.unlink()
    subprocess.run(
        [str(REPO / ".venv" / "bin" / "jupyter"), "trust", str(REPO / notebook)],
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
    return tuple(sum(p[i] for p in px) / n for i in range(3))


def run_probe(notebook: str, expected_n: int, slug: str, headed: bool, banner_timeout_ms: int) -> dict:
    port = find_free_port()
    print(f"[probe] starting jupyter on :{port}")
    jupyter = start_jupyter(port, notebook)
    result: dict = {"port": port, "notebook": notebook}
    chunk_progress: list[int] = []

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

            url = f"http://127.0.0.1:{port}/lab/tree/{notebook}?token={TOKEN}"
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

            # Poll the in-canvas status banner for chunk progress; collect a
            # short trace just to prove streaming actually happened.
            t0 = time.monotonic()
            seen_done = False
            while time.monotonic() - t0 < banner_timeout_ms / 1000:
                banner = page.evaluate(
                    """() => {
                        const s = document.querySelector('div[style*=\"border-left\"]');
                        return s ? s.textContent : '';
                    }"""
                )
                m = re.search(r"loading\s+(\d+)\s+/\s+(\d+)\s+chunks", banner or "")
                if m:
                    cur = int(m.group(1))
                    if not chunk_progress or chunk_progress[-1] != cur:
                        chunk_progress.append(cur)
                if banner and "FPS" in banner and "wall" in banner:
                    seen_done = True
                    break
                time.sleep(0.5)
            result["chunk_progress"] = chunk_progress
            result["banner_done"] = seen_done

            if not seen_done:
                print("[probe] never saw final banner")
                shot = REPO / "scripts" / f"p51b-{slug}-canvas.png"
                page.screenshot(path=str(shot), full_page=True)
                result["screenshot"] = str(shot)
                return result

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            print_out = next(
                (o for o in outputs if "render_mode" in o and "FPS" in o), ""
            )
            result["print_out"] = print_out[:400]
            # Pull the final banner's "across N chunks" report from the
            # canvas's status div (visible to us as 'banner_done' source).
            banner_final = page.evaluate(
                """() => {
                    const s = document.querySelector('div[style*=\"border-left\"]');
                    return s ? s.textContent : '';
                }"""
            )
            m_chunks = re.search(r"across\s+(\d+)\s+chunks", banner_final or "")
            if m_chunks:
                result["final_n_chunks"] = int(m_chunks.group(1))
            m = re.search(r"FPS\s*:\s*([\d.]+)", print_out)
            if m:
                result["fps"] = float(m.group(1))
            m = re.search(r"frame_ms\s*:\s*([\d.]+)", print_out)
            if m:
                result["frame_ms"] = float(m.group(1))
            m = re.search(r"rows\s*:\s*([\d,]+)", print_out)
            if m:
                result["rows"] = int(m.group(1).replace(",", ""))
            m = re.search(r"render_mode\s*:\s*([\w-]+)", print_out)
            if m:
                result["render_mode"] = m.group(1)
            m = re.search(r"bytes_recv\s*:\s*([\d.]+)\s*MB", print_out)
            if m:
                result["bytes_recv_mb"] = float(m.group(1))

            canvas = page.locator(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]'
            ).first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            shot = REPO / "scripts" / f"p51b-{slug}-canvas.png"
            canvas.screenshot(path=str(shot))
            result["screenshot"] = str(shot)
            result["rgb_center"] = tuple(round(v, 1) for v in avg_region_rgb(shot, 0.5, 0.5))
            result["rgb_corner"] = tuple(round(v, 1) for v in avg_region_rgb(shot, 0.05, 0.05))

            browser.close()

    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", choices=("synthetic", "tpch"), default="synthetic")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    if args.notebook == "synthetic":
        nb = "examples/p51b_100m_smoke.ipynb"
        expected_n = 100_000_000
        slug = "synth100m"
        timeout = 600_000
    else:
        nb = "examples/p51b_tpch_smoke.ipynb"
        expected_n = 149_996_355
        slug = "tpch"
        timeout = 900_000

    r = run_probe(nb, expected_n, slug, args.headed, timeout)

    print("\n" + "=" * 60)
    print(f"P5.1b PROBE — {args.notebook}")
    for k in (
        "render_mode", "rows", "fps", "frame_ms", "bytes_recv_mb",
        "chunk_progress", "banner_done",
        "rgb_center", "rgb_corner", "screenshot",
    ):
        if k in r:
            print(f"  {k:<16}: {r[k]}")
    print("=" * 60)

    mode_ok = r.get("render_mode") == "density-only"
    rows_ok = r.get("rows") == expected_n
    fps_ok = (r.get("fps") or 0) > 50.0
    chunks_observed = (r.get("final_n_chunks") or 0) > 1 or len(r.get("chunk_progress") or []) > 1
    center_bright = sum(r.get("rgb_center") or (0, 0, 0)) > 60.0
    corner_dark = sum(r.get("rgb_corner") or (255, 255, 255)) < 90.0
    done_ok = bool(r.get("banner_done"))

    print(f"  mode == density-only: {mode_ok}")
    print(f"  rows == expected ({expected_n:,}): {rows_ok}")
    print(f"  FPS > 50: {fps_ok}")
    print(f"  chunked transport observed: {chunks_observed} (chunks={r.get('final_n_chunks')})")
    print(f"  finalize banner observed: {done_ok}")
    print(f"  center has density pixels: {center_bright}")
    print(f"  corner is empty: {corner_dark}")

    ok = mode_ok and rows_ok and fps_ok and chunks_observed and done_ok and center_bright and corner_dark
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
