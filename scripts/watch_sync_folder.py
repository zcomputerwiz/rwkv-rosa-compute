#!/usr/bin/env python3
"""Event-driven watcher + wake bridge for the shared sync directory.

Deployed by opencode-dijkstra per D:\\ProjectSync\\WATCHER_SETUP_GUIDE.md
(author: antigravity-ampere). Local modifications:
  - IGNORE_NAMES customized to this node's own published files.
  - Continuous operation: does not exit on first event; keeps scanning until
    the shift timeout so subsequent events wake promptly.
  - Wake bridge: on a detected event, invokes `opencode run` headlessly so a
    fresh agent turn digests the change (debounced, single-flight).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WATCH_DIR = Path("C:/ProjectSync") if Path("C:/ProjectSync").exists() else Path("D:/ProjectSync")
WORKSPACE = Path("D:/OpenCode")
WAKE_LOG_DIR = WORKSPACE / "wake_runs"
STATE_FILE = WORKSPACE / "watch_state.json"
DEBOUNCE_SEC = 300

IGNORE_NAMES = {
    ".stfolder",
    ".stversions",
    ".tmp",
    # --- opencode-dijkstra's own publications (do not self-trigger) ---
    "AGENT_OPENCODE_DIJKSTRA.md",
    "JOB_STATUS_OPENCODE_DIJKSTRA.md",
}

WAKE_PROMPT = (
    "Automated watchdog turn - label output 'watchdog', never sign as "
    "dijkstra. A change summary has already been computed for you. Use the "
    "Read tool on D:\\OpenCode\\pending_digest.md, then output a digest of "
    "at most 15 lines: one line per changed item (name - author if "
    "recognizable - one-line point), a line asking-for-work if any item "
    "addresses opencode-dijkstra, and verbatim any TRANSFERRING/UNVERIFIED "
    "rows. Do not run any other tools. If the file contains only a header, "
    "output exactly 'Nothing new.'"
)


def build_pending_digest(summary: list[str]) -> None:
    """Precompute everything the wake turn needs, deterministically."""
    lines = [f"# digest generated {datetime.now().isoformat(timespec='seconds')}",
             "# replace this header line with 'PROCESSED' after reading"]
    for entry in summary:
        p = Path(entry.split(" ", 2)[-1].rsplit(" (", 1)[0]) \
            if entry.startswith(("CREATED", "MODIFIED")) else None
        status = ""
        if p is not None and p.suffix != ".sha256":
            side = p.with_suffix(p.suffix + ".sha256")
            if not side.exists():
                status = "TRANSFERRING (no sidecar yet)"
            else:
                h = hashlib.sha256()
                try:
                    with open(p, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b""):
                            h.update(chunk)
                    ok = h.hexdigest() == side.read_text(
                        encoding="utf-8").split()[0]
                    status = "READY" if ok else "TRANSFERRING (hash mismatch)"
                except OSError:
                    status = "UNREADABLE"
        lines.append(f"- {entry}{(' -> ' + status) if status else ''}")
    out = WORKSPACE / "pending_digest.md"
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(out)


def load_baseline() -> dict | None:
    try:
        return {k: tuple(v) for k, v in json.loads(
            STATE_FILE.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return None


def save_baseline(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass

WAKE_MODEL = "opencode/nemotron-3-ultra-free"


def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part in {".stfolder", ".stversions", "__pycache__"} or part.endswith(".tmp") or part.startswith("~") or part.startswith(".sync-conflict-"):
            return True
    if path.name in IGNORE_NAMES or path.name.endswith(".tmp") or path.name.startswith("~"):
        return True
    return False


def scan_state(root: Path) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    if not root.exists():
        return state
    try:
        for p in root.rglob("*"):
            if not is_ignored(p) and p.is_file():
                try:
                    st = p.stat()
                    state[str(p)] = (st.st_mtime_ns, st.st_size)
                except OSError:
                    pass
    except OSError:
        pass
    return state


def last_wake_age() -> float:
    marker = WORKSPACE / ".last_wake_ts"
    try:
        return time.time() - float(marker.read_text())
    except (OSError, ValueError):
        return float("inf")


def stamp_wake() -> None:
    try:
        (WORKSPACE / ".last_wake_ts").write_text(str(time.time()))
    except OSError:
        pass


def resolve_opencode() -> str | None:
    """Find a runnable opencode CLI. Scheduler contexts lack our PATH."""
    import shutil
    cand = os.environ.get("OPENCODE_EXE")
    if cand and Path(cand).exists():
        return cand
    found = shutil.which("opencode")
    if found:
        return found
    for fixed in (
        r"C:\Users\zcomp\AppData\Roaming\npm\opencode.cmd",
        r"C:\Users\zcomp\AppData\Local\Programs\@opencode-aidesktop\OpenCode.exe",
    ):
        if Path(fixed).exists():
            return fixed
    return None


def notify_operator(title: str, body: str) -> None:
    """Windows toast so landings are visible outside logs and sessions."""
    if os.environ.get("SYNC_TOAST", "1") != "1":
        return
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    ps = f"""
$ErrorActionPreference='SilentlyContinue'
[void][Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]
[void][Windows.UI.Notifications.ToastNotification,Windows.UI.Notifications,ContentType=WindowsRuntime]
$app='{{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}}\\WindowsPowerShell\\v1.0\\powershell.exe'
$xml=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t=$xml.GetElementsByTagName('text')
[void]$t.Item(0).AppendChild($xml.CreateTextNode('{esc(title)}'))
[void]$t.Item(1).AppendChild($xml.CreateTextNode('{esc(body)}'))
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""
    try:
        import base64
        enc = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-EncodedCommand", enc],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass


def wake_agent(summary: list[str]) -> None:
    age = last_wake_age()
    if age < DEBOUNCE_SEC:
        print(f"Wake debounced ({age:.0f}s < {DEBOUNCE_SEC}s); "
              f"{len(summary)} changes deferred.", flush=True)
        return
    exe = resolve_opencode()
    if exe is None:
        print("Wake dispatch FAILED: no opencode executable found "
              "(env OPENCODE_EXE unset, not on PATH, fallbacks missing).",
              flush=True)
        return
    try:
        build_pending_digest(summary)
    except Exception as exc:
        print(f"digest build failed: {exc!r}", flush=True)
    WAKE_LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = WAKE_LOG_DIR / f"wake_{stamp}.log"
    try:
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write("EVENTS:\n" + "\n".join(f"  - {s}" for s in summary) + "\n\n")
            fh.flush()
            subprocess.Popen(
                [exe, "run", "-m", WAKE_MODEL, "--title", "watchdog-sync",
                 WAKE_PROMPT],
                stdout=fh, stderr=subprocess.STDOUT,
                cwd=str(WORKSPACE),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        stamp_wake()
        n_files = sum(1 for s in summary if not s.endswith(".sha256)"))
        notify_operator(
            "Sync folder activity",
            f"{n_files} new file(s) landed - watchdog digest dispatched",
        )
        print(f"Wake dispatched via {exe} -> {outfile.name}", flush=True)
    except OSError as exc:
        print(f"Wake dispatch failed ({exe}): {exc}", flush=True)


def main() -> int:
    shift_sec = int(sys.argv[1]) if len(sys.argv) > 1 else 14400
    while True:  # self-restarting shifts; this process is the only writer
        print(f"Monitoring {WATCH_DIR} for incoming agent events "
              f"(shift {shift_sec}s)...", flush=True)
        try:
            run_shift(shift_sec)
        except Exception as exc:  # never die; log and continue
            print(f"Shift crashed: {exc!r}", flush=True)
            time.sleep(5)


def run_shift(shift_sec: int) -> None:
    # Persisted baseline (claudecode-v2 idea): restarts never silently absorb
    # changes that landed while nothing was armed.
    initial_state = load_baseline() or scan_state(WATCH_DIR)
    start_time = time.time()

    while (time.time() - start_time) < shift_sec:
        time.sleep(1.0)
        current_state = scan_state(WATCH_DIR)

        changed = []
        for path_str, (mtime, size) in current_state.items():
            if path_str not in initial_state:
                changed.append(f"CREATED: {path_str} ({size} bytes)")
            elif initial_state[path_str] != (mtime, size):
                changed.append(f"MODIFIED: {path_str} ({size} bytes)")
        for path_str in initial_state:
            if path_str not in current_state:
                changed.append(f"DELETED: {path_str}")

        if changed:
            print("\n" + "=" * 60, flush=True)
            print(f"EVENT DETECTED in {WATCH_DIR}:", flush=True)
            for c in changed:
                print(f"  - {c}", flush=True)
            print("=" * 60, flush=True)

            # Settle period for multi-file transfers
            time.sleep(2.0)
            build_pending_digest(changed)
            wake_agent(changed)

            initial_state = current_state
            save_baseline(current_state)


if __name__ == "__main__":
    raise SystemExit(main())
