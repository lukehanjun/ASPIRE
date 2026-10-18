#!/usr/bin/env python3
"""Replay an unchanged baseline policy through CaP-X with no LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--robospoon-root", type=Path, default=Path("/workspace/hanjun/robospoon"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.robospoon_root / "capx-baseline"))
    from capx.integrations.motion import pyroki
    from capx.integrations.vision import graspnet, sam3

    sam3.SERVICE_URL = "http://127.0.0.1:8234"
    graspnet.SERVICE_URL = "http://127.0.0.1:8235"
    pyroki.init_pyroki.__defaults__ = ("http://127.0.0.1:8236",)
    pyroki.init_pyroki_trajopt.__defaults__ = ("http://127.0.0.1:8236",)
    from capx.envs.simulators.libero import FrankaLiberoEnv
    from capx.integrations.franka.libero_reduced_skill_library import (
        FrankaLiberoApiReducedSkillLibrary,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = FrankaLiberoEnv(
        "libero_goal_swap", args.task_id, privileged=False,
        max_steps=8000, enable_render=True, viser_debug=False,
    )
    try:
        api = FrankaLiberoApiReducedSkillLibrary(env)
        env.reset(seed=args.seed)
        state = np.asarray(env.handle.env.sim.get_state().flatten())
        np.save(output / "initial-state.npy", state, allow_pickle=False)
        namespace = api.functions()
        text = args.policy.read_text()
        blocks = [s for s in re.split(r"^# Code block \d+[^\n]*\n", text, flags=re.MULTILINE) if s.strip()]
        errors = []
        started = time.monotonic()
        for index, block in enumerate(blocks):
            try:
                exec(compile(block, str(args.policy) + f":block-{index}", "exec"), namespace)
            except Exception:
                error = traceback.format_exc()
                errors.append({"block": index, "traceback": error})
                print(error, flush=True)
        final_state = np.asarray(env.handle.env.sim.get_state().flatten())
        np.save(output / "final-state.npy", final_state, allow_pickle=False)
        Image.fromarray(env.render()).save(output / "final.png")
        report = {
            "scope": "Unchanged generated policy replay through RoboSpoon CaP-X S3 simulator/API with ASPIRE frozen runtime/assets and original providers",
            "model_calls": 0, "included_in_baseline_score": False,
            "task_id": args.task_id, "seed": args.seed,
            "policy": str(args.policy.resolve()),
            "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "task_completed": bool(env.task_completed()),
            "simulator_steps": env._sim_step_count,
            "elapsed_seconds": time.monotonic() - started,
            "code_blocks": len(blocks), "execution_errors": errors,
        }
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.handle.env.close()


if __name__ == "__main__":
    main()
