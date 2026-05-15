"""Probe: load demo_fineweb_paint.ipynb at 10M, lasso, recolor, assert."""

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
TOKEN = "stipple-probe-token"
NOTEBOOK = "examples/demo_fineweb_paint.ipynb"


def find_free_port(start: int = 9350) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def wait_for_jupyter(port: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api?token={TOKEN}", timeout=1
            ) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("jupyter never started")


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


def main() -> int:
    port = find_free_port()
    print(f"[probe] :{port}")
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
            ctx = browser.new_context(viewport={"width": 1280, "height": 860})
            page = ctx.new_page()
            page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))
            url = f"http://127.0.0.1:{port}/lab/tree/{NOTEBOOK}?token={TOKEN}"
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_selector(".jp-Notebook", state="attached", timeout=60_000)
            time.sleep(2.0)

            # Run All
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
            page.fill(".lm-CommandPalette-input", "Run All Cells")
            time.sleep(0.3)
            page.keyboard.press("Enter")

            page.wait_for_selector(
                'canvas[data-stipple-role="render"]', state="attached", timeout=120_000
            )
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /FPS:\\s*[\\d.]+/.test(o.textContent));
                }""",
                timeout=300_000,
            )
            print("[probe] density loaded")

            # Capture a baseline of the canvas (uniform color).
            page.locator('canvas[data-stipple-role="render"]').first.scroll_into_view_if_needed()
            time.sleep(0.5)
            (REPO / "scripts" / "probe_paint_before.png").write_bytes(page.screenshot())

            # Lasso a small region in the middle.
            box = page.locator('canvas[data-stipple-role="render"]').first.bounding_box()
            cx = box["x"] + box["width"] * 0.55
            cy = box["y"] + box["height"] * 0.45
            import math
            radius = min(box["width"], box["height"]) * 0.12

            page.keyboard.down("Shift")
            page.mouse.move(cx + radius, cy)
            page.mouse.down()
            for i in range(1, 33):
                ang = (i / 32) * 2 * math.pi
                page.mouse.move(cx + radius * math.cos(ang), cy + radius * math.sin(ang))
                time.sleep(0.020)
            page.mouse.move(cx + radius, cy)
            page.mouse.up()
            page.keyboard.up("Shift")

            # Wait for selection_count to bump.
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /lasso:\\s*[\\d,]+/.test(o.textContent));
                }""",
                timeout=20_000,
            )
            # Dump the full widget output so we can see how many points
            # the lasso actually captured.
            widget_out = page.evaluate(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    const wid = outs.find(o => /lasso:\\s*[\\d,]+/.test(o.textContent));
                    return wid ? wid.textContent : '';
                }"""
            )
            print(f"[probe] lasso landed; widget output:\n  {widget_out[-500:]!r}")
            # Give Python time to assemble chunked selection.
            time.sleep(2.5)

            # Capture cell execution counts before re-run so we can
            # verify the right cell ran.
            before_counts = page.evaluate(
                """() => {
                    const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                    return cells.map(c => {
                        const p = c.querySelector('.jp-InputPrompt');
                        return p ? p.textContent.trim() : '';
                    });
                }"""
            )
            print(f"[probe] cell prompts before re-run: {before_counts}")

            # Focus the paint cell, then run via Shift+Enter.
            paint_cell = page.locator(".jp-CodeCell", has_text="w.update_color(sims)").first
            paint_cell.scroll_into_view_if_needed()
            time.sleep(0.5)
            paint_cell.locator(".cm-editor").first.click()
            time.sleep(0.4)
            page.keyboard.press("Escape")
            time.sleep(0.3)
            active_src = page.evaluate(
                """() => {
                    const active = document.querySelector('.jp-Cell.jp-mod-active');
                    return active ? (active.querySelector('.cm-content')?.textContent || '').slice(0, 80) : 'NONE';
                }"""
            )
            print(f"[probe] active cell source: {active_src!r}")
            page.keyboard.press("Shift+Enter")
            time.sleep(1.5)

            after_counts = page.evaluate(
                """() => {
                    const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                    return cells.map(c => {
                        const p = c.querySelector('.jp-InputPrompt');
                        return p ? p.textContent.trim() : '';
                    });
                }"""
            )
            print(f"[probe] cell prompts after re-run:  {after_counts}")

            # Wait for the cell's stdout confirming repaint.
            try:
                page.wait_for_function(
                    """() => {
                        const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                        const t = cells.find(c =>
                            (c.querySelector('.cm-content')?.textContent || '').includes('w.update_color(sims)')
                        );
                        if (!t) return false;
                        const out = t.querySelector('.jp-OutputArea-output');
                        return out && /repainted in one GPU upload/.test(out.textContent);
                    }""",
                    timeout=30_000,
                )
                print("[probe] python ack received")
            except Exception as e:
                # Dump every code cell's output for diagnosis.
                debug = page.evaluate(
                    """() => {
                        const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                        return cells.map((c, i) => {
                            const src = (c.querySelector('.cm-content')?.textContent || '').slice(0, 60);
                            const out = c.querySelector('.jp-OutputArea-output');
                            return `[${i}] src=${JSON.stringify(src)} out=${JSON.stringify(out ? out.textContent.slice(0, 200) : null)}`;
                        }).join('\\n');
                    }"""
                )
                print("[probe] code cells:")
                print(debug)
                (REPO / "scripts" / "probe_paint_timeout.png").write_bytes(page.screenshot())
                print(f"[probe] saved timeout screenshot")
                raise

            # Give the GPU a tick to actually swap buffers + render.
            time.sleep(1.0)
            page.locator('canvas[data-stipple-role="render"]').first.scroll_into_view_if_needed()
            (REPO / "scripts" / "probe_paint_after.png").write_bytes(page.screenshot())

            # Diff before vs after — expect a LARGE change.
            from PIL import Image, ImageChops
            import io
            import numpy as np
            before = Image.open(REPO / "scripts" / "probe_paint_before.png").convert("RGB")
            after = Image.open(REPO / "scripts" / "probe_paint_after.png").convert("RGB")
            diff = np.array(ImageChops.difference(before, after), dtype=np.int32).mean()
            print(f"[probe] before/after pixel diff: {diff:.2f}")
            print(
                "[probe] PASS" if diff > 3.0 else
                "[probe] FAIL: canvas didn't visibly recolor"
            )

            ctx.close()
            browser.close()
    finally:
        jupyter.terminate()
        try:
            jupyter.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
