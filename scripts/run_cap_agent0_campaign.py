#!/usr/bin/env python3
"""Supervise the ten-task released CaP-Agent0 baseline and write live scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def summarize(root, active):
    results = root / "trial-results.jsonl"
    rows = {}
    duplicates = []
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "/evaluation/" not in row.get("output_dir", ""):
                continue
            key = (row["task_id"], row["trial"])
            if key in rows:
                duplicates.append(list(key))
            rows[key] = row
    tasks = []
    all_tasks = json.loads((root / "tasks.json").read_text())
    for task in all_tasks:
        completed = [row for (task_id, seed), row in rows.items() if task_id == task["task_id"]]
        passed = sum(row.get("task_completed") is True for row in completed)
        tasks.append({
            "task_id": task["task_id"], "instruction": task["language"],
            "completed": len(completed), "expected": 15, "successes": passed,
            "success_rate": passed / len(completed) if completed else None,
        })
    successes = sum(row.get("task_completed") is True for row in rows.values())
    missing = [[task_id, seed] for task_id in range(10) for seed in range(51, 66)
               if (task_id, seed) not in rows]
    budget = root / "transport/budget.json"
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "running" if active else "complete" if not missing and not duplicates else "incomplete",
        "model": "claude-haiku-5-5", "suite": "libero_goal_swap", "seeds": list(range(51, 66)),
        "completed": len(rows), "expected": 150, "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "final_success_rate": successes / 150 if not missing and not duplicates else None,
        "tasks": tasks, "missing": missing, "duplicate_task_seeds": duplicates,
        "active_gpu_groups": active,
        "api_accounting": json.loads(budget.read_text()) if budget.exists() else None,
    }
    write_json(root / "summary.json", summary)
    lines = [
        f"CaP-Agent0 / Haiku 5.5 — {summary['status']}",
        f"Completed: {len(rows)}/150; successes: {successes}.",
        "Seeds: 51–65 for each of the ten libero_goal_swap tasks.", "",
        "Task | Completed | Successes | Rate among completed trials",
        "--- | --- | --- | ---",
    ]
    for task in tasks:
        rate = f"{task['success_rate']:.1%}" if task["success_rate"] is not None else "—"
        lines.append(f"{task['task_id']} {task['instruction']} | {task['completed']}/15 | {task['successes']} | {rate}")
    (root / "progress.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers-per-task", type=int, default=5)
    args = parser.parse_args()
    root = args.run_root.resolve()
    marker = root / "campaign-started.json"
    if marker.exists():
        parser.error("campaign already started; do not duplicate completed trials")
    groups = {0: [0, 3, 6, 9], 3: [1, 4, 7], 7: [2, 5, 8]}
    write_json(marker, {
        "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "groups": groups, "workers_per_task": args.workers_per_task,
        "launcher_sha256": hashlib.sha256((REPO / "scripts/run_cap_agent0.py").read_bytes()).hexdigest(),
        "gateway_sha256": hashlib.sha256((REPO / "scripts/cap_agent0_gateway.py").read_bytes()).hexdigest(),
    })
    processes = {}
    logs = []
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        for child in processes.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for gpu, task_ids in groups.items():
        env = os.environ.copy()
        env.update(
            ASPIRE_CAP_AGENT0_RUN_ROOT=str(root), ASPIRE_SIM_GPU=str(gpu),
            CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
            PYGLET_HEADLESS_DEVICE=str(gpu), PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        )
        for variable in ("OPENAI_API_KEY", "VECTOR_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(variable, None)
        log = (root / "logs" / f"gpu-{gpu}.log").open("w")
        logs.append(log)
        command = [
            str(REPO / ".venv-cap-agent0/bin/python"), str(REPO / "scripts/run_cap_agent0.py"),
            "--run-root", str(root), "--task-ids", *map(str, task_ids),
            "--workers", str(args.workers_per_task),
        ]
        child = subprocess.Popen(
            command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes[gpu] = child
        print(json.dumps({"gpu": gpu, "task_ids": task_ids, "pid": child.pid}), flush=True)
    last_completed = -1
    while True:
        active = [gpu for gpu, child in processes.items() if child.poll() is None]
        summary = summarize(root, active)
        if stopped:
            summary["status"] = "stopped"
            write_json(root / "summary.json", summary)
        if summary["completed"] != last_completed:
            print(json.dumps({k: summary[k] for k in ("status", "completed", "expected", "successes")}), flush=True)
            last_completed = summary["completed"]
        write_json(root / "processes.json", {
            str(gpu): {"pid": child.pid, "returncode": child.poll()} for gpu, child in processes.items()
        })
        if not active:
            break
        time.sleep(30)
    for log in logs:
        log.close()
    gateway_pid = root / "gateway.pid"
    if gateway_pid.exists():
        try:
            os.kill(int(gateway_pid.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
