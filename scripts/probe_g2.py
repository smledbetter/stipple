"""G2 verification probe.

Renders 1M GMM points colored by 3 cluster labels, measures FPS, exercises
mouse wheel zoom + drag pan, captures a canvas screenshot.

Kill criterion (per gate spec): FPS < 10 on M4 at 1280×800 means the
pipeline port is wrong. Target: FPS > 30.
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
TOKEN = "stipple-g2-token"
NOTEBOOK = "examples/g2_smoke.ipynb"


def find_free_port(start: int = 8920) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8920")


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
    # Use a probe-local config + workspaces dir so JupyterLab can't
    # accidentally restore a multi-tab session from a prior run.
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
            ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
            page = ctx.new_page()
            page.on("console", lambda m: print(f"[console] {m.type}: {m.text}"))
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))

            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            page.wait_for_selector(".jp-Notebook", state="attached", timeout=60_000)
            page.wait_for_selector(".jp-Cell", state="attached", timeout=30_000)
            time.sleep(3.0)

            # Focus the notebook via JS, then drive via keyboard. Avoids
            # Playwright's visibility heuristic which is flaky on JupyterLab 4.5.
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
                # Last-ditch: click the Run menu by accessible name
                page.get_by_role("menuitem", name="Run").first.click(force=True)
                page.get_by_role("menuitem", name="Run All Cells").first.click(force=True)

            print("[probe] waiting for 'Stipple G2' status with FPS…")
            t0 = time.monotonic()
            try:
                page.wait_for_function(
                    """() => {
                        const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                        return outs.some(o => /Stipple G2 — [\\d,]+ rows/.test(o.textContent)
                                              && /FPS: [\\d.]+/.test(o.textContent));
                    }""",
                    timeout=180_000,
                )
                wall = time.monotonic() - t0
                result["render_status"] = "success"
                result["wall_seconds"] = wall
                print(f"[probe] G2 rendered + benchmarked ({wall:.1f}s)")
            except Exception as e:
                print(f"[probe] WARNING: G2 render+benchmark not observed — {e}")
                result["render_status"] = "timeout"

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            result["outputs"] = outputs

            widget_text = next((o for o in outputs if "Stipple G2 — " in o), "")
            m_fps = re.search(r"FPS: ([\d.]+)", widget_text)
            if m_fps:
                result["fps"] = float(m_fps.group(1))
            m_frame = re.search(r"frame ([\d.]+) ms", widget_text)
            if m_frame:
                result["frame_ms"] = float(m_frame.group(1))
            m_rows = re.search(r"— ([\d,]+) rows", widget_text)
            if m_rows:
                result["rows"] = int(m_rows.group(1).replace(",", ""))

            # Pre-interaction canvas snapshot
            shot_pre = REPO / "scripts" / "g2-canvas-pre.png"
            try:
                canvas = page.locator(".jp-OutputArea-output canvas").first
                canvas.scroll_into_view_if_needed(timeout=5_000)
                time.sleep(0.3)
                canvas.screenshot(path=str(shot_pre))
                result["canvas_pre"] = str(shot_pre)
            except Exception as e:
                print(f"[probe] pre canvas snap failed: {e}")

            # Exercise wheel zoom + drag pan
            try:
                box = canvas.bounding_box()
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    page.mouse.move(cx, cy)
                    for _ in range(5):
                        page.mouse.wheel(0, -120)
                        time.sleep(0.05)
                    page.mouse.move(cx, cy)
                    page.mouse.down()
                    page.mouse.move(cx + 80, cy + 40, steps=10)
                    page.mouse.up()
                    time.sleep(0.5)
                    result["interaction"] = "ok"
            except Exception as e:
                print(f"[probe] interaction failed: {e}")
                result["interaction"] = f"failed: {e}"

            shot_post = REPO / "scripts" / "g2-canvas-post.png"
            try:
                canvas.screenshot(path=str(shot_post))
                result["canvas_post"] = str(shot_post)
            except Exception as e:
                print(f"[probe] post canvas snap failed: {e}")

            page_shot = REPO / "scripts" / "g2-result.png"
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
    print(f"G2 PROBE — render_status={r.get('render_status')}")
    if "wall_seconds" in r:
        print(f"  wall          : {r['wall_seconds']:.2f}s")
    if "rows" in r:
        print(f"  rows          : {r['rows']:,}")
    if "fps" in r:
        print(f"  fps           : {r['fps']:.1f}")
    if "frame_ms" in r:
        print(f"  frame_ms      : {r['frame_ms']:.2f}")
    if "interaction" in r:
        print(f"  interaction   : {r['interaction']}")
    print("=" * 60)
    fps = r.get("fps") or 0
    ok = r.get("render_status") == "success" and fps > 30
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
