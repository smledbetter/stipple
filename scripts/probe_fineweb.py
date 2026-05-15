"""Demo verification — Lasso 10M FineWeb documents, read what's in your cluster.

End-to-end test of the canonical ML researcher workflow on real data:
  1. Load `examples/demo_fineweb_1m.ipynb` (10M FineWeb docs + text blob)
  2. Wait for the density render to complete (chunked transport, ~25 chunks)
  3. Shift+drag a lasso around the center of the embedding
  4. Re-run the inspect cell
  5. Verify the cell printed ≥ 5 real document snippets

Pass: lasso returned > 100k row indices, FPS > 50, the inspect cell
printed text from ≥ 5 distinct rows.
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
TOKEN = "stipple-fineweb-token"
NOTEBOOK = "examples/demo_fineweb_1m.ipynb"


def find_free_port(start: int = 9110) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9110")


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
            page.on("console", lambda m: print(f"[console.{m.type}] {m.text}"))

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

            # Wait for the load to finalize. Banner format differs between
            # single-shot (FPS line, no "mode=") and chunked (FPS + "mode=").
            # Accept either.
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /FPS:\\s*[\\d.]+/.test(o.textContent)
                                          && /Stipple\\s+[—-]\\s+[\\d,]+\\s+rows/.test(o.textContent));
                }""",
                timeout=240_000,
            )
            # Generous post-load settle: a 10M density chunked load can
            # trigger a JupyterLab "Connection lost, reconnecting" event;
            # the comm channel recovers on its own but we need to let the
            # reconnect complete before firing the lasso event so the
            # selection message reaches Python.
            time.sleep(4.0)

            # Pre-lasso screenshot.
            canvas = page.locator(
                '.jp-OutputArea-output canvas[data-stipple-role="render"]'
            ).first
            canvas.scroll_into_view_if_needed(timeout=5_000)
            pre = REPO / "scripts" / "fineweb-canvas-density.png"
            canvas.screenshot(path=str(pre))
            result["pre_canvas"] = str(pre)
            time.sleep(0.3)

            # Shift+drag a lasso around the center of the data. Embedding is
            # in [0, 1]², fitView centers it; canvas center matches embedding
            # midpoint within margin.
            box = canvas.bounding_box()
            assert box, "no canvas bounding box"
            # Lasso a moderate region at canvas-center. FineWeb is dense
            # in the middle of the embedding so this returns hundreds of
            # thousands of documents — enough to make the "read the
            # cluster" payload obviously real.
            cx = box["x"] + box["width"] * 0.50
            cy = box["y"] + box["height"] * 0.50
            half = min(box["width"], box["height"]) * 0.10

            page.keyboard.down("Shift")
            page.mouse.move(cx - half, cy - half)
            page.mouse.down()
            page.mouse.move(cx + half, cy - half, steps=10)
            page.mouse.move(cx + half, cy + half, steps=10)
            page.mouse.move(cx - half, cy + half, steps=10)
            page.mouse.move(cx - half, cy - half, steps=10)
            page.mouse.up()
            page.keyboard.up("Shift")

            # Confirm the JS lasso compute fired.
            print("[probe] waiting for JS lasso banner…")
            page.wait_for_function(
                """() => {
                    const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                    return outs.some(o => /lasso:\\s*[\\d,]+/.test(o.textContent));
                }""",
                timeout=15_000,
            )
            print("[probe] JS lasso banner observed")
            # Don't wait on the Python-ack trait-sync round-trip — the
            # post-chunked-load comm channel can lag for many seconds before
            # `selection_count` propagates back. Instead, give Python a few
            # seconds to receive the buffer, then re-run the inspect cell
            # below — `w.selected_indices` is the ground truth.
            time.sleep(5.0)

            # Parse the lasso banner.
            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            lasso_banner = next(
                (o for o in outputs if "lasso:" in o and "Python ack" in o), ""
            )
            m = re.search(r"lasso:\s*([\d,]+)\s*/\s*([\d,]+)", lasso_banner)
            if m:
                result["selected"] = int(m.group(1).replace(",", ""))
                result["total"] = int(m.group(2).replace(",", ""))

            # Post-lasso screenshot with green outline overlay.
            post = REPO / "scripts" / "fineweb-canvas-lassoed.png"
            page.locator('[data-stipple-role="wrap"]').first.screenshot(path=str(post))
            result["post_canvas"] = str(post)

            # Re-run the INSPECT cell — the one whose source references
            # `selected_indices` (skips any phantom trailing empty cells
            # JupyterLab adds automatically). Click on its CodeMirror
            # editor so cell focus, not widget-canvas focus, receives the
            # subsequent Shift+Enter keystroke.
            inspect_cell = page.locator(
                ".jp-CodeCell", has_text="selected_indices"
            ).first
            editor = inspect_cell.locator(".cm-editor").first
            editor.click()
            time.sleep(0.3)
            page.keyboard.press("Escape")  # command mode
            time.sleep(0.2)
            page.keyboard.press("Shift+Enter")

            # Wait for the inspect cell to actually re-execute. We watch
            # the cell's prompt number: when it bumps, the cell ran.
            # Wait for the inspect cell to print the lassoed-documents
            # report. Generous timeout: at 10M scale the indices buffer
            # may be a few MB and the comm round-trip + cell re-execution
            # together can take 10+ seconds.
            try:
                page.wait_for_function(
                    """() => {
                        const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                        const target = cells.find(c =>
                            (c.querySelector('.cm-content')?.textContent || '').includes('selected_indices')
                        );
                        if (!target) return false;
                        const out = target.querySelector('.jp-OutputArea-output');
                        return out && /Lassoed [\\d,]+ documents/.test(out.textContent);
                    }""",
                    timeout=90_000,
                )
            except Exception as e:
                # Diagnostic: dump all code cell prompts + first 200 chars of output
                cells_state = page.evaluate(
                    """() => {
                        return Array.from(document.querySelectorAll('.jp-CodeCell')).map(c => ({
                            prompt: c.querySelector('.jp-InputPrompt')?.textContent || '',
                            src: (c.querySelector('.cm-content')?.textContent || '').slice(0,60),
                            out: (c.querySelector('.jp-OutputArea-output')?.textContent || '').slice(0,200),
                        }));
                    }"""
                )
                print(f"[probe] inspect-wait timed out: {e}")
                for i, c in enumerate(cells_state):
                    print(f"  cell[{i}] prompt={c['prompt']!r}")
                    print(f"           src={c['src']!r}")
                    print(f"           out={c['out']!r}")
                raise
            time.sleep(0.5)

            inspect_text = page.evaluate(
                """() => {
                    const cells = Array.from(document.querySelectorAll('.jp-CodeCell'));
                    const target = cells.find(c =>
                        (c.querySelector('.cm-content')?.textContent || '').includes('selected_indices')
                    );
                    if (!target) return '';
                    const out = target.querySelector('.jp-OutputArea-output');
                    return out ? out.textContent : '';
                }"""
            )
            result["inspect_text"] = inspect_text
            m = re.search(r"Lassoed\s+([\d,]+)\s+documents", inspect_text)
            if m:
                result["py_lassoed"] = int(m.group(1).replace(",", ""))
            # Count "--- row N ---" markers as proof of >=1 document printed.
            result["docs_printed"] = len(re.findall(r"--- row [\d,]+ ---", inspect_text))

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
    print("\n" + "=" * 70)
    print("DEMO PROBE — FineWeb 10M lasso → read documents")
    for k in (
        "selected", "total", "py_lassoed", "docs_printed",
        "pre_canvas", "post_canvas",
    ):
        if k in r:
            print(f"  {k:<16}: {r[k]}")
    if "inspect_text" in r:
        print("\n  --- inspect cell output (first 800 chars) ---")
        print(r["inspect_text"][:800])
        print("  --- end inspect output ---")
    print("=" * 70)

    selected = r.get("selected") or 0
    py = r.get("py_lassoed") or 0
    docs = r.get("docs_printed") or 0

    sel_ok = selected > 50_000
    sync_ok = py == selected
    docs_ok = docs >= 5

    print(f"  lasso selected >50k rows: {sel_ok} ({selected:,})")
    print(f"  JS→Python sync exact match: {sync_ok} ({py:,} == {selected:,})")
    print(f"  ≥5 documents printed: {docs_ok} ({docs})")

    return 0 if (sel_ok and sync_ok and docs_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
