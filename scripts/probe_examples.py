"""Drive the example notebooks (embeddings.ipynb, dedup.ipynb, g3_smoke.ipynb)
through Playwright. For each:

- launch / reuse Jupyter
- open the notebook
- run all cells
- wait for the widget's 'Stipple — N rows' banner
- (interactive notebooks only) simulate a shift+drag lasso, verify 'Python ack'
- capture a canvas screenshot

Exit 0 iff every notebook crosses its render gate.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

REPO = Path(__file__).resolve().parent.parent
TOKEN = "stipple-examples-token"


@dataclass
class NotebookSpec:
    name: str
    path: str
    expected_n: int
    do_lasso: bool
    # CSS-px coords as a fraction of the canvas bounding box to center the lasso
    lasso_cx_frac: float = 0.5
    lasso_cy_frac: float = 0.5
    lasso_half_frac: float = 0.12


SPECS: list[NotebookSpec] = [
    NotebookSpec(
        name="embeddings",
        path="examples/embeddings.ipynb",
        expected_n=1797,
        do_lasso=True,
        # sklearn digits PCA: cluster '0' tends to sit in the upper-left of the
        # default PCA projection; use a generous mid-left lasso to grab a couple
        # of digit classes.
        lasso_cx_frac=0.35,
        lasso_cy_frac=0.50,
        lasso_half_frac=0.20,
    ),
    NotebookSpec(
        name="dedup",
        path="examples/dedup.ipynb",
        expected_n=200_000,
        do_lasso=True,
        # The dedup notebook plants 50 micro-clusters at random in N(0, 3^2)^2.
        # Lasso a fat chunk around the origin to catch both noise + a few
        # planted clusters.
        lasso_cx_frac=0.50,
        lasso_cy_frac=0.50,
        lasso_half_frac=0.25,
    ),
    NotebookSpec(
        name="g3_smoke",
        path="examples/g3_smoke.ipynb",
        expected_n=10_000_000,
        do_lasso=False,  # just render + benchmark; lasso already covered
    ),
]


@dataclass
class Result:
    name: str
    rendered: bool = False
    rows: int = 0
    fps: float = 0.0
    decode_ms: float = 0.0
    first_render_ms: float = 0.0
    lasso_selected: int = 0
    lasso_gpu_ms: float = 0.0
    python_ack: bool = False
    canvas_path: str = ""
    notes: list[str] = field(default_factory=list)

    def ok(self, spec: NotebookSpec) -> bool:
        if not self.rendered:
            return False
        if self.rows != spec.expected_n:
            return False
        if spec.do_lasso and not self.python_ack:
            return False
        return True


def find_free_port(start: int = 8960) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port near 8960")


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
    # Pre-trust the example notebooks
    for spec in SPECS:
        subprocess.run(
            [str(REPO / ".venv" / "bin" / "jupyter"), "trust", str(REPO / spec.path)],
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
            "--ServerApp.tornado_settings={\"websocket_max_message_size\": 524288000}",
            "--ServerApp.max_buffer_size=536870912",
        ],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def run_one(page: Page, spec: NotebookSpec, port: int) -> Result:
    r = Result(name=spec.name)
    url = f"http://127.0.0.1:{port}/lab/tree/{spec.path}?token={TOKEN}"
    print(f"\n[{spec.name}] navigating: {spec.path}")
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

    print(f"[{spec.name}] waiting for 'Stipple — {spec.expected_n:,} rows' banner…")
    # Pass the exact comma-formatted count; the JS side strips commas before
    # testing so we don't have to fight regex escaping.
    expected_n_str = f"{spec.expected_n:,}"
    try:
        page.wait_for_function(
            "(want) => {"
            "  const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));"
            "  return outs.some(o => {"
            "    const t = o.textContent;"
            "    return t.indexOf(`Stipple \\u2014 ${want} rows`) !== -1 && /FPS: \\d+/.test(t);"
            "  });"
            "}",
            arg=expected_n_str,
            timeout=180_000,
        )
        r.rendered = True
        print(f"[{spec.name}] render banner observed")
    except Exception as e:
        # Dump cell contents so we can see what actually rendered
        outs_dbg = page.locator(".jp-OutputArea-output").all_text_contents()
        print(f"[{spec.name}] WARNING: render timeout. Cell outputs:")
        for i, o in enumerate(outs_dbg):
            preview = (o[:300] + "…") if len(o) > 300 else o
            print(f"  [{i}] {preview!r}")
        r.notes.append(f"render timeout: {e}")
        return r

    # Parse banner
    outs = page.locator(".jp-OutputArea-output").all_text_contents()
    banner = next((o for o in outs if "Stipple — " in o), "")
    m_rows = re.search(r"Stipple — ([\d,]+) rows", banner)
    if m_rows:
        r.rows = int(m_rows.group(1).replace(",", ""))
    m_fps = re.search(r"FPS: ([\d.]+)", banner)
    if m_fps:
        r.fps = float(m_fps.group(1))
    m_dec = re.search(r"decode: ([\d.]+) ms", banner)
    if m_dec:
        r.decode_ms = float(m_dec.group(1))
    m_fr = re.search(r"first render: ([\d.]+) ms", banner)
    if m_fr:
        r.first_render_ms = float(m_fr.group(1))

    # Canvas snapshot
    try:
        canvas = page.locator(".jp-OutputArea-output canvas").first
        canvas.scroll_into_view_if_needed(timeout=5_000)
        time.sleep(0.4)
        shot = REPO / "scripts" / f"ex-{spec.name}-canvas.png"
        canvas.screenshot(path=str(shot))
        r.canvas_path = str(shot)
    except Exception as e:
        r.notes.append(f"canvas screenshot failed: {e}")

    if not spec.do_lasso:
        return r

    # Simulate shift+drag lasso
    try:
        box = canvas.bounding_box()
        assert box, "no canvas bounding box"
        cx = box["x"] + box["width"] * spec.lasso_cx_frac
        cy = box["y"] + box["height"] * spec.lasso_cy_frac
        half = min(box["width"], box["height"]) * spec.lasso_half_frac
        print(f"[{spec.name}] lassoing around ({cx:.0f}, {cy:.0f}) ±{half:.0f}px")

        page.keyboard.down("Shift")
        page.mouse.move(cx - half, cy - half)
        page.mouse.down()
        page.mouse.move(cx + half, cy - half, steps=8)
        page.mouse.move(cx + half, cy + half, steps=8)
        page.mouse.move(cx - half, cy + half, steps=8)
        # Mid-drag screenshot: polyline is visible on the overlay canvas before
        # the closing pointerup clears it.
        try:
            shot_mid = REPO / "scripts" / f"ex-{spec.name}-lasso-mid.png"
            canvas.screenshot(path=str(shot_mid))
        except Exception:
            pass
        page.mouse.move(cx - half, cy - half, steps=8)
        page.mouse.up()
        page.keyboard.up("Shift")

        page.wait_for_function(
            """() => {
                const outs = Array.from(document.querySelectorAll('.jp-OutputArea-output'));
                return outs.some(o => /lasso: [\\d,]+ \\/ .*Python ack ✓/.test(o.textContent));
            }""",
            timeout=15_000,
        )
        outs2 = page.locator(".jp-OutputArea-output").all_text_contents()
        b2 = next((o for o in outs2 if "Python ack" in o), "")
        m_l = re.search(
            r"lasso: ([\d,]+) / [\d,]+ selected · gpu ([\d.]+) ms · Python ack",
            b2,
        )
        if m_l:
            r.lasso_selected = int(m_l.group(1).replace(",", ""))
            r.lasso_gpu_ms = float(m_l.group(2))
            r.python_ack = True

        # Reliably re-run the last cell via JupyterLab's command palette
        # (clicking the cell + keyboard shortcut is flaky in headless).
        try:
            # Focus the last cell first
            page.evaluate(
                """() => {
                    const cells = document.querySelectorAll('.jp-Notebook .jp-Cell');
                    const last = cells[cells.length - 1];
                    if (last) {
                        last.scrollIntoView({ block: 'center' });
                        last.click();
                    }
                }"""
            )
            time.sleep(0.4)
            # Esc to ensure command mode, then trigger Run Cell via palette
            page.keyboard.press("Escape")
            time.sleep(0.2)
            page.keyboard.press("Meta+Shift+C")
            page.wait_for_selector(".lm-CommandPalette-input", timeout=5_000)
            page.fill(".lm-CommandPalette-input", "Run Selected Cells")
            time.sleep(0.4)
            page.keyboard.press("Enter")
            time.sleep(2.0)
            outs3 = page.locator(".jp-OutputArea-output").all_text_contents()
            print(f"[{spec.name}] post-rerun outputs:")
            for i, o in enumerate(outs3[-3:]):
                preview = (o[:250] + "…") if len(o) > 250 else o
                print(f"    [{i}] {preview!r}")
            for o in outs3:
                if ("Selected " in o and " samples" in o) or (
                    "Selected " in o and " points" in o
                ):
                    r.notes.append(f"inspection: {o.strip()[:300]}")
                    break
            else:
                r.notes.append(
                    "inspection re-run did not produce a 'Selected ...' line"
                )
        except Exception as e:
            r.notes.append(f"inspection re-run failed: {e}")
    except Exception as e:
        r.notes.append(f"lasso failed: {e}")
        print(f"[{spec.name}] WARNING: lasso failed: {e}")

    return r


def main() -> int:
    headed = "--headed" in sys.argv
    port = find_free_port()
    print(f"[probe] starting jupyter on :{port}")
    jp = start_jupyter(port)
    results: list[Result] = []

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

            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            # Only surface meaningful console errors (mute the benign
            # "widget model not found" probe-during-setup error)
            page.on(
                "console",
                lambda m: (
                    errors.append(f"console {m.type}: {m.text}")
                    if m.type == "error" and "widget model not found" not in m.text
                    else None
                ),
            )

            for spec in SPECS:
                results.append(run_one(page, spec, port))

            browser.close()
            if errors:
                print("\n[probe] page errors:")
                for e in errors[-20:]:
                    print(f"  {e}")

    finally:
        jp.terminate()
        try:
            jp.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jp.kill()

    print("\n" + "=" * 70)
    print("EXAMPLES PROBE")
    print("=" * 70)
    any_fail = False
    for spec, r in zip(SPECS, results):
        status = "PASS" if r.ok(spec) else "FAIL"
        any_fail |= not r.ok(spec)
        print(
            f"  {status}  {spec.name:<11}  rows={r.rows:>10,}  "
            f"fps={r.fps:>5.1f}  decode={r.decode_ms:>5.1f}ms  "
            f"first={r.first_render_ms:>5.1f}ms"
        )
        if spec.do_lasso:
            print(
                f"           lasso={r.lasso_selected:>10,}  "
                f"gpu={r.lasso_gpu_ms:>5.1f}ms  ack={r.python_ack}"
            )
        for n in r.notes:
            print(f"           note: {n}")
        if r.canvas_path:
            print(f"           canvas: {r.canvas_path}")
    print("=" * 70)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
