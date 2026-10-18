#!/usr/bin/env python3
"""Compare CaP-X reset states under an independently selected Python runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 65])
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    sys.path.insert(0, str(args.source_root / "capx-baseline"))
    from capx.envs.simulators.libero import FrankaLiberoEnv
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    versions = {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "torch", "robosuite")}
    for task_id in range(10):
        env = FrankaLiberoEnv(
            "libero_goal_swap", task_id, privileged=False,
            max_steps=8000, enable_render=True, viser_debug=False,
        )
        try:
            for seed in args.seeds:
                env.reset(seed=seed)
                state = np.asarray(env.handle.env.sim.get_state().flatten())
                np.save(args.output / f"task-{task_id}-seed-{seed}.npy", state, allow_pickle=False)
                other = np.load(args.reference_root / "host-audit/evaluation" / f"task_{task_id}" / f"seed_{seed}" / "initial-state-0.npy", allow_pickle=False)
                row = {
                    "task_id": task_id, "seed": seed, "state_shape": list(state.shape),
                    "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    "matches_aspire_state": bool(np.array_equal(state, other)),
                    "maximum_state_difference": float(np.max(np.abs(state - other))),
                    "qpos_maximum_difference": float(np.max(np.abs(state[1:1 + env.handle.env.sim.model.nq] - other[1:1 + env.handle.env.sim.model.nq]))),
                }
                rows.append(row)
                report = {
                    "model_calls": 0, "python_executable": sys.executable,
                    "versions": versions, "asset_config": "ASPIRE shared LIBERO_CONFIG_PATH",
                    "reference_root": str(args.reference_root), "tasks": rows,
                }
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
        finally:
            env.handle.env.close()


if __name__ == "__main__":
    main()
