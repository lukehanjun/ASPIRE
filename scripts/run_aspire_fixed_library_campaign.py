#!/usr/bin/env python3
"""Coordinate isolated native Haiku task workers and frozen-program evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--after-robospoon", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    protocol = json.loads((root / "campaign.json").read_text())
    marker = root / "campaign-started.json"
    if marker.exists():
        parser.error("campaign already started; never duplicate model workers or evaluation")
    groups = {0: [0, 3, 6, 9], 3: [1, 4, 7], 7: [2, 5, 8]}
    write_json(marker, {"started_at": now(), "pid": os.getpid(), "groups": groups})
    python = "/workspace/hanjun/ASPIRE/.venv-cap-agent0/bin/python"
    state = {"status": "waiting-for-robospoon", "started_at": now(), "tasks": {}, "gpu_ledger": {}, "successes": 0, "evaluated_trials": 0, "expected_trials": 150}
    lock = threading.RLock()
    children = {}
    stopped = threading.Event()

    def save():
        state["updated_at"] = now()
        state["successes"] = sum(t.get("successes", 0) for t in state["tasks"].values())
        state["evaluated_trials"] = sum(t.get("evaluation_trials", 0) for t in state["tasks"].values())
        state["final_success_rate"] = state["successes"] / 150 if state["evaluated_trials"] == 150 else None
        write_json(root / "summary.json", state)
        lines = [f"ASPIRE fixed empty library / Haiku 5.5 — {state['status']}", f"Held-out: {state['evaluated_trials']}/150; native successes: {state['successes']}.", "Development66–80; evaluation51–65. No reusable skill learning or cross-task transfer.", "", "Task | State | Development | Evaluation", "--- | --- | --- | ---"]
        for tid in range(10):
            task = state["tasks"].get(str(tid), {})
            lines.append(f"{tid} | {task.get('state', 'pending')} | {task.get('development_successes', '—')}/15 | {task.get('successes', '—')}/{task.get('evaluation_trials', '—')}")
        (root / "progress.md").write_text("\n".join(lines) + "\n")

    def stop(signum, frame):
        stopped.set()
        with lock:
            for child in children.values():
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for started in (root / "tasks").glob("task_*/*/*/started.json"):
                pid = int(json.loads(started.read_text())["pid"])
                command = Path(f"/proc/{pid}/cmdline")
                try:
                    if str(root).encode() in command.read_bytes():
                        os.kill(pid, signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with lock:
        save()
    while not stopped.is_set():
        previous = args.after_robospoon / "summary.json"
        if previous.exists():
            previous_state = json.loads(previous.read_text())
            if previous_state.get("status") == "complete":
                break
            if previous_state.get("status") in ("incomplete", "stopped"):
                state.update(status="blocked-by-incomplete-robospoon", dependency_summary=str(previous))
                save()
                return 1
        stopped.wait(30)
    if stopped.is_set():
        state["status"] = "stopped"
        save()
        return 1
    with (args.after_robospoon / "logs/final-validation.log").open("w") as log:
        validation = subprocess.run(
            [python, str(root / "configs/report_cap_agent_comparison.py"), "--run-root", str(args.after_robospoon)],
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if validation.returncode:
        state["status"] = "blocked-by-robospoon-validation"
        save()
        return 1
    state["status"] = "running"
    save()

    def run_process(command, env, log_path, key):
        if stopped.is_set():
            return -1
        with log_path.open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            with lock:
                children[key] = child
            while child.poll() is None:
                if stopped.wait(20):
                    return -1
                with lock:
                    save()
            with lock:
                children.pop(key, None)
            return child.returncode

    def run_gpu(gpu, task_ids):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), PYGLET_HEADLESS_DEVICE=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        for key in ("OPENAI_API_KEY", "VECTOR_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY", "ASPIRE_CAP_AGENT0_RUN_ROOT", "ROBOSPOON_CAP_AGENT0_RUN_ROOT"):
            env.pop(key, None)
        for tid in task_ids:
            if stopped.is_set():
                return
            task_dir = root / f"tasks/task_{tid}"
            with lock:
                state["tasks"][str(tid)] = {"state": "stage1-running", "gpu": gpu, "started_at": now()}
                state["gpu_ledger"][str(gpu)] = {"task_id": tid, "phase": "development"}
                save()
            returncode = run_process([
                python, str(root / "configs/run_vector_claude.py"),
                "--prompt-file", str(task_dir / "worker-prompt.md"), "--output-dir", str(task_dir / "worker/agent"),
                "--budget-usd", str(protocol["per_task_claude_code_budget_usd"]),
                "--timeout-minutes", str(protocol["worker_timeout_minutes"]),
                "--gateway-url", "http://127.0.0.1:8231", "--scope", f"aspire-task-{tid}", "--cwd", str(task_dir),
            ], env, task_dir / "logs/agent.log", f"worker-{tid}")
            if stopped.is_set():
                return
            # A native Bash background sweep may outlive the coding-agent turn.
            # Keep ownership until it finishes; never overlap two simulation jobs.
            for attempt in task_dir.glob("*/*/started.json"):
                pid = int(json.loads(attempt.read_text())["pid"])
                while alive(pid) and not stopped.wait(20):
                    with lock:
                        save()
            if stopped.is_set():
                return
            candidates = []
            for results_path in (task_dir / "development").glob("*/results.json"):
                rows = json.loads(results_path.read_text())
                if sorted(row["seed"] for row in rows) == protocol["development_seeds"]:
                    program = results_path.parent / "program.py"
                    candidates.append((sum(r["task_completed"] for r in rows), sum(r["program_ok"] for r in rows), -program.stat().st_size, results_path.parent.name, program))
            if candidates:
                passed, valid, _, version, program = max(candidates)
                shutil.copy2(program, task_dir / "fix_code.py")
            elif (task_dir / "fix_code.py").exists():
                passed, version = None, "worker-fallback-no-complete-development-sweep"
            else:
                with lock:
                    state["tasks"][str(tid)].update(state="missing-program", worker_returncode=returncode)
                    save()
                continue
            frozen = task_dir / "fix_code.py"
            digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
            handoff = {"task_id": tid, "state": "stage1-done", "selected_version": version, "development_successes": passed, "development_trials": 15 if candidates else None, "code_sha256": digest, "code_path": str(frozen), "worker_returncode": returncode, "frozen_at": now()}
            write_json(task_dir / "stage1_result.json", handoff)
            with lock:
                state["tasks"][str(tid)].update(handoff, state="evaluation-running")
                state["gpu_ledger"][str(gpu)] = {"task_id": tid, "phase": "evaluation"}
                save()
            evaluation_env = {**env, "ASPIRE_COORDINATOR_EVALUATION": "1"}
            version = "frozen-" + digest[:12]
            returncode = run_process([
                python, str(root / "configs/run_aspire_fixed_library_program.py"), "--campaign-root", str(root),
                "--task-id", str(tid), "--seeds", *map(str, protocol["evaluation_seeds"]),
                "--program", str(frozen), "--version", version, "--phase", "evaluation",
            ], evaluation_env, task_dir / "logs/evaluation.log", f"evaluation-{tid}")
            results_path = task_dir / "evaluation" / version / "results.json"
            rows = json.loads(results_path.read_text()) if results_path.exists() else []
            valid = len(rows) == 15 and all(r["matches_aspire_initial_state"] and r["program_sha256"] == digest for r in rows)
            valid = valid and hashlib.sha256(frozen.read_bytes()).hexdigest() == digest
            with lock:
                state["tasks"][str(tid)].update(state="done" if valid and returncode == 0 else "evaluation-incomplete", evaluation_trials=len(rows), successes=sum(r["task_completed"] for r in rows), evaluation_returncode=returncode, evaluation_valid=valid, finished_at=now())
                state["gpu_ledger"][str(gpu)] = {"task_id": None, "phase": "free"}
                save()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_gpu, gpu, tasks) for gpu, tasks in groups.items()]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                with lock:
                    state.setdefault("infrastructure_errors", []).append(repr(exc))
                    save()
    with lock:
        state["status"] = "stopped" if stopped.is_set() else "complete" if all(state["tasks"].get(str(t), {}).get("state") == "done" for t in range(10)) else "incomplete"
        state["finished_at"] = now()
        save()
    try:
        os.kill(int((root / "gateway.pid").read_text()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    print(json.dumps(state), flush=True)
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
