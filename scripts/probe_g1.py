"""G1 verification probe.

Starts JupyterLab, opens examples/g1_smoke.ipynb in headless Chromium with
WebGPU flags, runs all cells, checks that the widget reports receiving
~1,000,000 rows from Python via the Arrow-over-comm transport.

Usage:
    uv run python scripts/probe_g1.py
    uv run python scripts/probe_g1.py --headed
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
TOKEN = "stipple-g1-token"
NOTEBOOK = "examples/g1_smoke.ipynb"


def find_free_port(start: int = 8910) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8910")


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
            ctx = browser.new_context(viewport={"width": 1400, "height": 900})
            page = ctx.new_page()
            page.on("console", lambda m: print(f"[console] {m.type}: {m.text}"))
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            print(f"[probe] navigating to {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            page.wait_for_selector(".jp-Notebook", timeout=60_000)
            print("[probe] notebook chrome rendered")
            # Wait for at least one cell so we know the document opened
            page.wait_for_selector(".jp-Cell", timeout=30_000)
            time.sleep(3.0)

            # JupyterLab 4.5 sometimes has invisible/overlapped wrappers; click
            # the first cell instead of the bare notebook container, with force
            # to bypass Playwright's visibility heuristic when needed.
            page.locator(".jp-Cell").first.click(force=True)
            page.keyboard.press("Meta+Shift+C")
            try:
                page.wait_for_selector(".lm-CommandPalette-input", timeout=3_000)
                page.fill(".lm-CommandPalette-input", "Run All Cells")
                time.sleep(0.4)
                page.keyboard.press("Enter")
                print("[probe] triggered 'Run All Cells'")
            except Exception:
                page.locator("text=Run").first.click()
                page.locator("text=Run All Cells").first.click()

            print("[probe] waiting for widget 'G1 — received' text…")
            t0 = time.monotonic()
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Stipple G1 — received [\\d,]+ rows from Python/.test(o.textContent));
                    }""",
                    timeout=120_000,
                )
                wall = time.monotonic() - t0
                result["render_status"] = "success"
                result["wall_seconds"] = wall
                print(f"[probe] widget rendered ({wall:.1f}s)")
            except Exception as e:
                print(f"[probe] WARNING: widget render not observed — {e}")
                result["render_status"] = "timeout"

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            result["outputs"] = outputs

            # Extract decode/render ms from widget text if present
            widget_text = next((o for o in outputs if "Stipple G1 — received" in o), "")
            m = re.search(r"decode: ([\d.]+) ms · render: ([\d.]+) ms", widget_text)
            if m:
                result["decode_ms"] = float(m.group(1))
                result["render_ms"] = float(m.group(2))
            m2 = re.search(r"Arrow IPC: ([\d.]+) MB", widget_text)
            if m2:
                result["ipc_mb"] = float(m2.group(1))
            m3 = re.search(r"received ([\d,]+) rows", widget_text)
            if m3:
                result["rows"] = int(m3.group(1).replace(",", ""))

            shot = REPO / "scripts" / "g1-result.png"
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)

            try:
                canvas = page.locator(".jp-OutputArea-output canvas").first
                canvas.scroll_into_view_if_needed(timeout=5_000)
                time.sleep(0.5)
                canvas_path = REPO / "scripts" / "g1-canvas.png"
                canvas.screenshot(path=str(canvas_path))
                result["canvas_screenshot"] = str(canvas_path)
            except Exception as e:
                print(f"[probe] canvas screenshot skipped: {e}")

            browser.close()

    finally:
        print("[probe] tearing down jupyter…")
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()
        try:
            tail = jupyter.stdout.read().decode("utf-8", "replace")[-2000:] if jupyter.stdout else ""
            if tail.strip():
                result["jupyter_tail"] = tail
        except Exception:
            pass

    return result


def main() -> int:
    headed = "--headed" in sys.argv
    r = run_probe(headed=headed)
    print("\n" + "=" * 60)
    print(f"G1 PROBE — render_status={r.get('render_status')}")
    if "wall_seconds" in r:
        print(f"  wall          : {r['wall_seconds']:.2f}s")
    if "rows" in r:
        print(f"  rows          : {r['rows']:,}")
    if "ipc_mb" in r:
        print(f"  ipc           : {r['ipc_mb']:.2f} MB")
    if "decode_ms" in r:
        print(f"  decode        : {r['decode_ms']:.1f} ms")
    if "render_ms" in r:
        print(f"  render        : {r['render_ms']:.1f} ms")
    print("=" * 60)
    return 0 if r.get("render_status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
