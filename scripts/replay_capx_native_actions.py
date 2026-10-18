#!/usr/bin/env python3
"""Compare native simulators with an identical action stream and no model calls."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-state", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--policy", type=Path)
    mode.add_argument("--actions", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root / "capx-baseline"), str(args.source_root)]
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import mujoco

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    low = FrankaLiberoEnv("libero_goal_swap", args.task_id, privileged=False, max_steps=8000)
    low.reset(seed=args.seed)
    native = low.handle.env
    if args.initial_state:
        native.set_init_state(np.load(args.initial_state, allow_pickle=False))
    initial = np.asarray(native.sim.get_state().flatten()).copy()
    if args.initial_state:
        assert np.array_equal(initial, np.load(args.initial_state, allow_pickle=False))
    np.save(output / "initial-state.npy", initial, allow_pickle=False)
    actions, states, outcomes, errors = [], [], [], []
    original_step = native.step

    def record_step(action):
        result = original_step(action)
        actions.append(np.asarray(action).copy())
        states.append(np.asarray(native.sim.get_state().flatten()).copy())
        outcomes.append(bool(low.task_completed()))
        return result

    native.step = record_step
    started = time.monotonic()
    try:
        if args.policy:
            from capx.integrations.motion import pyroki
            from capx.integrations.vision import graspnet, sam3

            sam3.SERVICE_URL = "http://127.0.0.1:8234"
            graspnet.SERVICE_URL = "http://127.0.0.1:8235"
            pyroki.init_pyroki.__defaults__ = ("http://127.0.0.1:8236",)
            pyroki.init_pyroki_trajopt.__defaults__ = ("http://127.0.0.1:8236",)
            from capx.integrations.franka.libero_reduced_skill_library import (
                FrankaLiberoApiReducedSkillLibrary,
            )

            namespace = FrankaLiberoApiReducedSkillLibrary(low).functions()
            blocks = [
                block
                for block in re.split(
                    r"^# Code block \d+[^\n]*\n", args.policy.read_text(), flags=re.MULTILINE
                )
                if block.strip()
            ]
            for index, block in enumerate(blocks):
                try:
                    exec(compile(block, str(args.policy) + f":block-{index}", "exec"), namespace)
                except Exception:
                    errors.append(traceback.format_exc())
        else:
            sequence = np.load(args.actions, allow_pickle=False)["actions"]
            for action in sequence:
                # BaseEnv.step is a no-motion fallback in released CaP-X.
                # Replay at the same native boundary used by its motion API.
                low.handle.step(action)
                low._sim_step_count += 1
        np.savez_compressed(
            output / "actions-and-states.npz",
            actions=np.asarray(actions),
            states=np.asarray(states),
            task_success=np.asarray(outcomes),
        )
        final = np.asarray(native.sim.get_state().flatten()).copy()
        report = {
            "scope": "zero-LLM fixed native action stream diagnostic; excluded from all success-rate evaluations",
            "task_id": args.task_id,
            "seed": args.seed,
            "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
            "action_stream_sha256": hashlib.sha256(np.asarray(actions).tobytes()).hexdigest(),
            "simulator_steps": low._sim_step_count,
            "native_actions": len(actions),
            "task_success_final": bool(low.task_completed()),
            "task_success_ever": any(outcomes),
            "final_state_sha256": hashlib.sha256(final.tobytes()).hexdigest(),
            "wall_s": time.monotonic() - started,
            "policy_errors": errors,
            "python": sys.version,
            "mujoco_version": mujoco.__version__,
            "source_root": str(args.source_root),
            "restored_exact_initial_state": str(args.initial_state) if args.initial_state else None,
        }
        if args.actions:
            reference = np.load(args.actions, allow_pickle=False)
            report["exact_action_match"] = bool(
                np.array_equal(reference["actions"], np.asarray(actions))
            )
            report["max_abs_state_delta_mixed_units"] = float(
                np.max(abs(reference["states"] - np.asarray(states)))
            )
            report["task_success_trace_identical"] = bool(
                np.array_equal(reference["task_success"], np.asarray(outcomes))
            )
            report["max_qpos_difference"] = float(
                np.max(
                    abs(
                        reference["states"][:, 1 : 1 + native.sim.model.nq]
                        - np.asarray(states)[:, 1 : 1 + native.sim.model.nq]
                    )
                )
            )
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        native.close()


if __name__ == "__main__":
    main()
