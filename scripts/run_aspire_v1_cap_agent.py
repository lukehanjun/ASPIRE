#!/usr/bin/env python3
"""Run fresh CaP-X agents using the frozen ASPIRE v1 API, without learning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

GATEWAY = "http://127.0.0.1:8232/v1/chat/completions"


def append_json(path, row):
    raw = (json.dumps(row, default=lambda value: value.item()) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, raw)
    finally:
        os.close(descriptor)


def install_runtime():
    root = Path(os.environ["ASPIRE_V1_CAP_AGENT_RUN_ROOT"])
    source = root / "source"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "capx-baseline"))
    from capx.envs import runner
    sys.path.insert(0, str(root / "configs"))
    sys.path.insert(0, str(source / "aspire-v1/src"))
    from capx.llm import client
    import numpy as np

    if "claude-haiku-5-5" not in client.VLM_MODELS:
        client.VLM_MODELS.append("claude-haiku-5-5")
    original = runner._run_trial_with_retries
    if getattr(original, "comparison_recording", False):
        return

    def recorded_trial(env, trial, args, config, multi_turn_prompt):
        low = env.low_level_env
        task_id = int(low.handle.task_id)
        scope = f"aspire-v1-capx:task-{task_id}:seed-{trial}"
        args.api_key = scope
        args.visual_differencing_model_api_key = scope
        args.visual_differencing_model_server_url = GATEWAY
        phase = "preflight" if "/preflight/" in str(config["output_dir"]) else "evaluation"
        env._phase = phase
        audit = root / "host-audit" / phase / f"task_{task_id}" / f"seed_{trial}"
        audit.mkdir(parents=True, exist_ok=True)
        reset = low.reset
        signatures = []
        matches = []
        reference_root = Path(json.loads((root / "baseline.json").read_text())["aspire_reference_root"])

        def reset_and_record(*reset_args, **reset_kwargs):
            result = reset(*reset_args, **reset_kwargs)
            state = np.asarray(low.handle.env.sim.get_state().flatten())
            np.save(audit / f"initial-state-{len(signatures)}.npy", state, allow_pickle=False)
            signatures.append(hashlib.sha256(state.tobytes()).hexdigest())
            reference = reference_root / "host-audit/evaluation" / f"task_{task_id}" / f"seed_{trial}" / "initial-state-0.npy"
            matches.append(bool(np.array_equal(state, np.load(reference, allow_pickle=False))))
            if not matches[-1]:
                raise RuntimeError(f"paired initial state mismatch: task {task_id} seed {trial}")
            return result

        low.reset = reset_and_record
        try:
            summary = original(env, trial, args, config, multi_turn_prompt)
        finally:
            low.reset = reset
        env.finalize(summary)
        record = asdict(summary)
        record.pop("log", None)
        record.update(
            suite="libero_goal_swap", task_id=task_id, model=args.model,
            initial_state_sha256=signatures[0] if signatures else None,
            retry_initial_states_sha256=signatures,
            matches_aspire_initial_state=matches[0] if matches else None,
            retry_matches_aspire_initial_state=matches,
            max_steps=low.max_steps, simulator_steps=low._sim_step_count,
            output_dir=str(config["output_dir"]), pid=os.getpid(),
            v1_artifact_dir=str(env.recorder.root), no_learning=True,
        )
        append_json(root / "trial-results.jsonl", record)
        print("BASELINE_TRIAL " + json.dumps(record, default=lambda value: value.item()), flush=True)
        return summary

    recorded_trial.comparison_recording = True
    runner._run_trial_with_retries = recorded_trial


if os.environ.get("ASPIRE_V1_CAP_AGENT_RUN_ROOT"):
    install_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--start-seed", type=int, default=51)
    parser.add_argument("--end-seed", type=int, default=65)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-tag", default="")
    args = parser.parse_args()
    if not all(0 <= task < 10 for task in args.task_ids) or not 1 <= args.start_seed <= args.end_seed:
        parser.error("invalid task or seed interval")
    root = args.run_root.resolve()
    os.environ["ASPIRE_V1_CAP_AGENT_RUN_ROOT"] = str(root)
    os.environ.pop("ASPIRE_CAP_AGENT0_RUN_ROOT", None)
    os.environ.pop("ROBOSPOON_CAP_AGENT0_RUN_ROOT", None)
    os.environ.pop("CAPX_ENV_STACK", None)
    for name in ("OPENAI_API_KEY", "VECTOR_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(name, None)
    install_runtime()
    from capx.envs.scripts import run_libero_batch as batch
    import yaml

    source = root / "source/capx-baseline"
    config = yaml.safe_load((source / "env_configs/libero/franka_libero_cap_agent0_noplay_s3.yaml").read_text())
    assert config["evolve_skill_library"] is False
    config["resume_idx"] = args.start_seed
    config["api_servers"] = []  # Already running, version-verified common providers.
    phase = "preflight" if args.preflight else "evaluation"
    if args.preflight_tag and (not args.preflight or not args.preflight_tag.replace("-", "").isalnum()):
        parser.error("preflight tags require --preflight and an alphanumeric tag")
    phase_root = root / phase
    if args.preflight_tag:
        phase_root /= args.preflight_tag
    config_path = root / "configs" / f"{phase}{args.preflight_tag}_{'_'.join(map(str, args.task_ids))}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # CaP-X's batch CLI omits the VDM URL override exposed by its LaunchArgs.
    launch = batch.launch_main

    def launch_with_vdm_transport(launch_args):
        launch_args.visual_differencing_model_server_url = GATEWAY
        path = Path(launch_args.config_path)
        generated = yaml.safe_load(path.read_text())
        generated["env"]["_target_"] = "aspire_v1_cap_agent_adapter.AspireV1CodeEnv"
        generated["env"]["cfg"]["low_level"]["_target_"] = "aspire_v1_cap_agent_adapter.AspireV1Low"
        generated["env"]["cfg"]["apis"] = []
        generated["evolve_skill_library"] = False
        path.write_text(yaml.safe_dump(generated, sort_keys=False))
        return launch(launch_args)

    batch.launch_main = launch_with_vdm_transport
    os.chdir(source)
    batch.main(batch.LiberoBatchLaunchArgs(
        base_config_path=str(config_path), suites=["libero_goal_swap"],
        task_ids=args.task_ids, models=["claude-haiku-5-5"], server_url=GATEWAY,
        output_dir=str(phase_root), temperature=1.0, max_tokens=20480,
        reasoning_effort="medium", total_trials=args.end_seed, num_workers=args.workers,
        record_video=True, visual_differencing_model="claude-haiku-5-5",
    ))


if __name__ == "__main__":
    main()
