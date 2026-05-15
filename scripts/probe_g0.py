"""G0 verification probe.

Starts JupyterLab, opens examples/g0_smoke.ipynb in headless Chromium with
WebGPU flags, runs all cells, checks that the widget's status trait flips
from 'checking' to 'success', and saves a canvas screenshot.

Usage:
    uv run python scripts/probe_g0.py           # headless
    uv run python scripts/probe_g0.py --headed
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
TOKEN = "stipple-g0-token"
NOTEBOOK = "examples/g0_smoke.ipynb"


def find_free_port(start: int = 8900) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8900")


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
            time.sleep(2.5)  # let kernel attach

            page.locator(".jp-Notebook").first.click()
            page.keyboard.press("Meta+Shift+C")
            try:
                page.wait_for_selector(".lm-CommandPalette-input", timeout=3_000)
                page.fill(".lm-CommandPalette-input", "Run All Cells")
                time.sleep(0.4)
                page.keyboard.press("Enter")
                print("[probe] triggered 'Run All Cells'")
            except Exception:
                print("[probe] command palette fallback — menu")
                page.locator("text=Run").first.click()
                page.locator("text=Run All Cells").first.click()

            # Wait for the widget's own status DOM to say it rendered.
            # The widget reports synchronously in its own output area; the
            # round-trip back to the Python kernel's trait is async, so a
            # separate print(w.status) cell will race.
            print("[probe] waiting for widget render text…")
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Stipple G0 — \\d+ points rendered/.test(o.textContent));
                    }""",
                    timeout=90_000,
                )
                result["render_status"] = "success"
                print("[probe] widget rendered")
            except Exception as e:
                print(f"[probe] WARNING: widget render not observed — {e}")
                result["render_status"] = "timeout"

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            result["outputs"] = outputs

            shot = REPO / "scripts" / "g0-result.png"
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)

            try:
                canvas = page.locator(".jp-OutputArea-output canvas").first
                canvas.scroll_into_view_if_needed(timeout=5_000)
                time.sleep(0.5)
                canvas_path = REPO / "scripts" / "g0-canvas.png"
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
    print(f"G0 PROBE — render_status={r.get('render_status')}")
    print("=" * 60)
    for i, out in enumerate(r.get("outputs", [])):
        print(f"\n--- output {i} ---")
        print(out[:1200])
    if "jupyter_tail" in r:
        print("\n--- jupyter tail ---")
        print(r["jupyter_tail"])
    print(f"\nscreenshot: {r.get('screenshot', '(none)')}")
    print(f"canvas    : {r.get('canvas_screenshot', '(none)')}")
    return 0 if r.get("render_status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
