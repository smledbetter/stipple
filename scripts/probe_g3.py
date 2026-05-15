"""G3 verification probe — 10M points.

Gate criterion: first render < 5s AND FPS > 15. Kill: first render > 5s OR
FPS < 5.

The widget reports decode/upload/first-render times in its status banner;
the probe pulls those plus the FPS benchmark and reports them.
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
TOKEN = "stipple-g3-token"
NOTEBOOK = "examples/g3_smoke.ipynb"


def find_free_port(start: int = 8930) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8930")


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
    jdir = REPO / ".jupyter-probe"
    (jdir / "lab" / "workspaces").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JUPYTER_DISABLE_BROWSER_OPEN"] = "1"
    env["JUPYTER_CONFIG_DIR"] = str(jdir)
    env["JUPYTERLAB_WORKSPACES_DIR"] = str(jdir / "lab" / "workspaces")
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
            "--LabApp.workspaces_dir=" + str(jdir / "lab" / "workspaces"),
            # Bump websocket message cap for 10M+ rows (default ~100MB).
            '--ServerApp.tornado_settings={"websocket_max_message_size": 524288000}',
            "--ServerApp.max_buffer_size=536870912",
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
            try:
                page.wait_for_selector(".lm-CommandPalette-input", timeout=5_000)
                page.fill(".lm-CommandPalette-input", "Run All Cells")
                time.sleep(0.4)
                page.keyboard.press("Enter")
            except Exception:
                page.get_by_role("menuitem", name="Run").first.click(force=True)
                page.get_by_role("menuitem", name="Run All Cells").first.click(force=True)

            print("[probe] waiting for G2-style banner with 10M rows + FPS…")
            t0 = time.monotonic()
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Stipple G2 — 10,000,000 rows/.test(o.textContent)
                                              && /FPS: [\\d.]+/.test(o.textContent));
                    }""",
                    timeout=240_000,
                )
                wall = time.monotonic() - t0
                result["render_status"] = "success"
                result["wall_seconds"] = wall
                print(f"[probe] G3 rendered + benchmarked ({wall:.1f}s)")
            except Exception as e:
                print(f"[probe] WARNING: G3 render not observed — {e}")
                result["render_status"] = "timeout"

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            result["outputs"] = outputs

            widget_text = next((o for o in outputs if "Stipple G2 — " in o), "")
            for key, pat in (
                ("rows", r"— ([\d,]+) rows"),
                ("ipc_mb", r"Arrow IPC: ([\d.]+) MB"),
                ("decode_ms", r"decode: ([\d.]+) ms"),
                ("upload_ms", r"upload: ([\d.]+) ms"),
                ("first_render_ms", r"first render: ([\d.]+) ms"),
                ("fps", r"FPS: ([\d.]+)"),
                ("frame_ms", r"frame ([\d.]+) ms"),
            ):
                m = re.search(pat, widget_text)
                if m:
                    v = m.group(1).replace(",", "")
                    result[key] = float(v) if "." in v else int(v)

            shot = REPO / "scripts" / "g3-canvas.png"
            try:
                canvas = page.locator(".jp-OutputArea-output canvas").first
                canvas.scroll_into_view_if_needed(timeout=5_000)
                time.sleep(0.5)
                canvas.screenshot(path=str(shot))
                result["canvas"] = str(shot)
            except Exception as e:
                print(f"[probe] canvas snap failed: {e}")

            page_shot = REPO / "scripts" / "g3-result.png"
            page.screenshot(path=str(page_shot), full_page=True)
            result["screenshot"] = str(page_shot)

            browser.close()

    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()
        try:
            tail = jupyter.stdout.read().decode("utf-8", "replace")[-1500:] if jupyter.stdout else ""
            if tail.strip():
                result["jupyter_tail"] = tail
        except Exception:
            pass

    return result


def main() -> int:
    headed = "--headed" in sys.argv
    r = run_probe(headed=headed)
    print("\n" + "=" * 60)
    print(f"G3 PROBE — render_status={r.get('render_status')}")
    for k in ("wall_seconds", "rows", "ipc_mb", "decode_ms", "upload_ms",
              "first_render_ms", "fps", "frame_ms"):
        if k in r:
            v = r[k]
            if isinstance(v, float):
                print(f"  {k:<16}: {v:.2f}")
            else:
                print(f"  {k:<16}: {v:,}" if isinstance(v, int) and v > 999 else f"  {k:<16}: {v}")
    print("=" * 60)

    fps = float(r.get("fps") or 0)
    first = float(r.get("first_render_ms") or 9999)
    ok = r.get("render_status") == "success" and fps > 15 and first < 5000
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
