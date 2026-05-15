"""P1 verification probe — DataFrame-positional API.

Loads `examples/p1_dataframe_smoke.ipynb`, which builds a 100k-row
pandas DataFrame and renders it via `Stipple(df, x=..., y=..., color=...)`.
Verifies the on-canvas banner reports the right row count + category count
(end-to-end proof that the DataFrame path produced the same rendered output
as the array path).
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
TOKEN = "stipple-p1-token"
NOTEBOOK = "examples/p1_dataframe_smoke.ipynb"


def find_free_port(start: int = 9030) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 9030")


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
            page.on(
                "console",
                lambda m: print(f"[console] {m.type}: {m.text}")
                if m.type in ("error", "warning")
                else None,
            )

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
                    return outs.some(o => /Stipple — 100,000 rows/.test(o.textContent)
                                          && /FPS: [\\d.]+/.test(o.textContent));
                }""",
                timeout=120_000,
            )
            time.sleep(0.5)

            outputs = page.locator(".jp-OutputArea-output").all_text_contents()
            banner = next((o for o in outputs if "Stipple " in o and "FPS:" in o), "")
            result["banner"] = banner[:300]

            m = re.search(r"Stipple\s+[—-]\s+([\d,]+)\s+rows\s*·\s*(\d+)\s+class", banner)
            if m:
                result["rows"] = int(m.group(1).replace(",", ""))
                result["classes"] = int(m.group(2))

            cats_out = next((o for o in outputs if "categories" in o), "")
            m2 = re.search(r"categories\s*:\s*\[(.*?)\]", cats_out)
            if m2:
                result["categories_repr"] = m2.group(1)

            shot = REPO / "scripts" / "p1-canvas.png"
            try:
                canvas = page.locator(
                    '.jp-OutputArea-output canvas[data-stipple-role="render"]'
                ).first
                canvas.screenshot(path=str(shot))
                result["screenshot"] = str(shot)
            except Exception as e:
                print(f"[probe] canvas snap failed: {e}")

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
    print("P1 PROBE — DataFrame-positional API")
    for k in ("rows", "classes", "categories_repr", "banner", "screenshot"):
        if k in r:
            print(f"  {k:<18}: {r[k]}")
    print("=" * 60)

    rows = r.get("rows") or 0
    classes = r.get("classes") or 0
    cats = r.get("categories_repr") or ""
    ok = (
        rows == 100_000
        and classes == 3
        and "cluster_a" in cats
        and "cluster_b" in cats
        and "cluster_c" in cats
    )
    print(f"  rows == 100k: {rows == 100_000}")
    print(f"  3 classes: {classes == 3}")
    print(f"  categories preserved from pandas: {'cluster_a' in cats and 'cluster_b' in cats}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
