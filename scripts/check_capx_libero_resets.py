#!/usr/bin/env python3
"""Check RoboSpoon CaP-X reset states in ASPIRE's frozen runtime, without an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--robospoon-root", type=Path, default=Path("/workspace/hanjun/robospoon"))
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    sys.path.insert(0, str(args.robospoon_root / "capx-baseline"))
    from capx.envs.simulators.libero import FrankaLiberoEnv

    root = args.run_root.resolve()
    output = root / "comparison/capx-reset-check"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id in range(10):
        env = FrankaLiberoEnv(
            "libero_goal_swap", task_id, privileged=False,
            max_steps=8000, enable_render=True, viser_debug=False,
        )
        try:
            env.reset(seed=args.seed)
            state = np.asarray(env.handle.env.sim.get_state().flatten())
            path = output / f"task-{task_id}-seed-{args.seed}.npy"
            np.save(path, state, allow_pickle=False)
            row = {
                "task_id": task_id, "seed": args.seed,
                "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                "state_path": str(path), "state_shape": list(state.shape),
                "simulator_steps": env._sim_step_count,
            }
            reference = root / f"host-audit/evaluation/task_{task_id}/seed_{args.seed}/initial-state-0.npy"
            if reference.exists():
                other = np.load(reference, allow_pickle=False)
                row["matches_completed_aspire_reset"] = bool(np.array_equal(state, other))
                row["maximum_state_difference"] = float(np.max(np.abs(state - other)))
            rows.append(row)
            (output / "results.json").write_text(json.dumps({
                "scope": "CaP-X S3 simulator source in ASPIRE's frozen v0.1 runtime/assets; not RoboSpoon's default runtime",
                "model_calls": 0, "tasks": rows,
            }, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        finally:
            env.handle.env.close()


if __name__ == "__main__":
    main()
