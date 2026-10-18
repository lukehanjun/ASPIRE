#!/usr/bin/env python3
"""Launch the released v0.1 CaP-Agent0 runner with Haiku and shared providers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "artifacts/releases/aspire-v0.1"
SIM = RELEASE / "aspire/sim"
sys.path.insert(0, str(RELEASE))


def append_json(path, row):
    raw = (json.dumps(row, default=lambda scalar: scalar.item()) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)


def install_runtime():
    from aspire.sim.cap.envs import runner
    from aspire.sim.cap.integrations.motion import pyroki
    from aspire.sim.cap.integrations.vision import graspnet, sam3
    from aspire.sim.cap.llm import client
    import numpy as np

    # Only transport/model registration changes. Keep original prompts, helper
    # implementation, reset, control, retry loop, and success scoring untouched.
    if "claude-haiku-5-5" not in client.VLM_MODELS:
        client.VLM_MODELS.append("claude-haiku-5-5")
    sam3.SERVICE_URL = "http://127.0.0.1:8234"
    graspnet.SERVICE_URL = "http://127.0.0.1:8235"
    pyroki.init_pyroki.__defaults__ = ("http://127.0.0.1:8236",)
    pyroki.init_pyroki_trajopt.__defaults__ = ("http://127.0.0.1:8236",)
    original = runner._run_trial_with_retries
    if getattr(original, "cap_agent0_recording", False):
        return

    def recorded_trial(env, trial, args, config, multi_turn_prompt):
        root = Path(os.environ["ASPIRE_CAP_AGENT0_RUN_ROOT"])
        low = env.low_level_env
        task_id = int(low.handle.task_id)
        scope = f"cap-agent0:task-{task_id}:seed-{trial}"
        args.api_key = scope  # local bookkeeping label; never an upstream credential
        args.visual_differencing_model_api_key = scope
        phase = "preflight" if "/preflight/" in str(config["output_dir"]) else "evaluation"
        audit = root / "host-audit" / phase / f"task_{task_id}" / f"seed_{trial}"
        audit.mkdir(parents=True, exist_ok=True)
        # The original trial resets *inside* its retry loop. Capture after that
        # reset, rather than the previous trial's state at this wrapper's entry.
        reset = low.reset
        signatures = []

        def reset_and_record(*reset_args, **reset_kwargs):
            result = reset(*reset_args, **reset_kwargs)
            native_state = np.asarray(low.handle.env.sim.get_state().flatten())
            np.save(audit / f"initial-state-{len(signatures)}.npy", native_state, allow_pickle=False)
            signatures.append(hashlib.sha256(native_state.tobytes()).hexdigest())
            return result

        low.reset = reset_and_record
        try:
            summary = original(env, trial, args, config, multi_turn_prompt)
        finally:
            low.reset = reset
        record = asdict(summary)
        record.pop("log", None)
        record.update(
            suite="libero_goal_swap", task_id=task_id, model=args.model,
            initial_state_sha256=signatures[0] if signatures else None,
            retry_initial_states_sha256=signatures, pid=os.getpid(),
            max_steps=low.max_steps, simulator_steps=low._sim_step_count,
            output_dir=str(config["output_dir"]),
        )
        append_json(root / "trial-results.jsonl", record)
        print("BASELINE_TRIAL " + json.dumps(record, default=lambda scalar: scalar.item()), flush=True)
        return summary

    recorded_trial.cap_agent0_recording = True
    runner._run_trial_with_retries = recorded_trial


# Multiprocessing spawn reimports this file. Reapply the same small transport
# configuration in each worker before the released runner creates its APIs.
if os.environ.get("ASPIRE_CAP_AGENT0_RUN_ROOT"):
    install_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--start-seed", type=int, default=51)
    parser.add_argument("--end-seed", type=int, default=65)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not all(0 <= task_id < 10 for task_id in args.task_ids):
        parser.error("task IDs must be in 0..9")
    if args.start_seed < 1 or args.end_seed < args.start_seed:
        parser.error("invalid seed interval")
    root = args.run_root.resolve()
    os.environ["ASPIRE_CAP_AGENT0_RUN_ROOT"] = str(root)
    for variable in ("OPENAI_API_KEY", "VECTOR_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(variable, None)
    install_runtime()
    from aspire.sim.cap.envs.scripts.run_libero_batch import LiberoBatchLaunchArgs
    from aspire.sim.cap.envs.scripts.run_libero_batch import main as batch
    import yaml

    tasks = json.loads((root / "tasks.json").read_text())
    config = yaml.safe_load((SIM / "env_configs/libero/franka_libero_baseline_image_diff_debug.yaml").read_text())
    config["resume_idx"] = args.start_seed
    prefix = "preflight" if args.preflight else "evaluation"
    slug = "_".join(map(str, args.task_ids))
    config_path = root / "configs" / f"{prefix}_{slug}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    os.chdir(SIM)
    batch(LiberoBatchLaunchArgs(
        base_config_path=str(config_path), suites=["libero_goal_swap"],
        models=["claude-haiku-5-5"],
        server_url="http://127.0.0.1:8230/v1/chat/completions",
        output_dir=str(root / prefix), temperature=1.0, max_tokens=20480,
        reasoning_effort="medium", total_trials=args.end_seed, num_workers=args.workers,
        record_video=True, visual_differencing_model="claude-haiku-5-5",
        visual_differencing_model_server_url="http://127.0.0.1:8230/v1/chat/completions",
        task_names=[task["task_name"] for task in tasks if task["task_id"] in args.task_ids],
    ))


if __name__ == "__main__":
    main()
