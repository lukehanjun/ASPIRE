#!/usr/bin/env python3
"""Run frozen programs with the paper-era ASPIRE API and matched LIBERO resets."""

from __future__ import annotations

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


class ProgramDeadline(BaseException):
    """Deadline that public API exception handlers must not suppress."""


def validate_program(text):
    tree = ast.parse(text)
    forbidden_names = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "vars", "globals", "locals", "APIS", "sim", "inner"}
    forbidden_attributes = {"sim", "data", "model", "_eval_predicate", "parsed_problem", "obj_body_id", "_step_once", "set_joint_qpos", "body_xpos", "get_site_xpos", "qpos", "__dict__", "__class__", "__globals__", "__builtins__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError(f"forbidden policy identifier: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr in forbidden_attributes or node.attr.startswith("_")):
            raise ValueError(f"forbidden policy attribute: {node.attr}")
        if isinstance(node, ast.Attribute) and node.attr == "env":
            raise ValueError("unwrapping the simulator is forbidden")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] not in {"math", "numpy", "scipy"} for name in modules):
                raise ValueError("policy imports are limited to math, numpy and scipy")


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--phase", choices=["development", "evaluation", "snapshot"], default="development")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    protocol = json.loads((root / "campaign.json").read_text())
    if not 0 <= args.task_id < 10 or len(args.seeds) != len(set(args.seeds)):
        parser.error("invalid task or duplicate seeds")
    allowed = protocol["evaluation_seeds"] if args.phase == "evaluation" else protocol["development_seeds"]
    if not set(args.seeds) <= set(allowed):
        parser.error("seed outside the assigned partition")
    if args.phase == "evaluation" and os.environ.get("ASPIRE_COORDINATOR_EVALUATION") != "1":
        parser.error("evaluation is coordinator-owned")
    if Path(args.version).name != args.version:
        parser.error("version must be a single path component")
    task_dir = root / f"tasks/task_{args.task_id}"
    output = task_dir / args.phase / args.version
    output.mkdir(parents=True, exist_ok=False)
    program = args.program.resolve() if args.program else None
    if args.phase != "snapshot" and (program is None or not program.is_relative_to(task_dir)):
        parser.error("program must be inside its assigned task directory")
    text = program.read_text() if program else ""
    validate_program(text)
    code_hash = hashlib.sha256(text.encode()).hexdigest()
    (output / "program.py").write_text(text)
    if args.phase == "development":
        counts = {}
        for previous in (task_dir / "development").glob("*/started.json"):
            for seed in json.loads(previous.read_text())["seeds"]:
                counts[seed] = counts.get(seed, 0) + 1
        if any(counts.get(seed, 0) >= protocol["max_development_attempts_per_seed"] for seed in args.seeds):
            parser.error("development replay limit reached")
    (output / "started.json").write_text(json.dumps({"started_at": now(), "seeds": args.seeds, "program_sha256": code_hash, "pid": os.getpid()}, indent=2) + "\n")
    for key in ("VECTOR_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(root / "source/aspire-v0.1"))
    from aspire.sim.cap.envs.simulators.libero import FrankaLiberoEnv
    from aspire.sim.cap.integrations.libero_trace_logger import (
        TracedFrankaLiberoApiReducedSkillLibrary,
    )
    from aspire.sim.cap.integrations.motion import pyroki
    from aspire.sim.cap.integrations.vision import graspnet, sam3
    import numpy as np
    from PIL import Image

    if not Path(inspect.getfile(FrankaLiberoEnv)).is_relative_to(root / "source"):
        raise RuntimeError("ASPIRE simulator did not import from the frozen source")
    sam3.SERVICE_URL = "http://127.0.0.1:8234"
    graspnet.SERVICE_URL = "http://127.0.0.1:8235"
    pyroki.init_pyroki.__defaults__ = ("http://127.0.0.1:8236",)
    pyroki.init_pyroki_trajopt.__defaults__ = ("http://127.0.0.1:8236",)
    low = FrankaLiberoEnv("libero_goal_swap", args.task_id, privileged=False, max_steps=8000, enable_render=True, viser_debug=False)
    api = TracedFrankaLiberoApiReducedSkillLibrary(low)
    reference = Path(protocol["aspire_reference_root"])
    rows = []

    def timed_out(signum, frame):
        raise ProgramDeadline("program exceeded 1000 seconds")

    signal.signal(signal.SIGALRM, timed_out)
    try:
        for seed in args.seeds:
            trial_dir = output / f"seed_{seed}"
            trial_dir.mkdir()
            low.reset(seed=seed)
            state = np.asarray(low.handle.env.sim.get_state().flatten())
            digest = hashlib.sha256(state.tobytes()).hexdigest()
            # Privileged native states remain in the host audit, outside task directories.
            audit = root / "host-audit" / args.phase / f"task_{args.task_id}" / args.version
            audit.mkdir(parents=True, exist_ok=True)
            np.save(audit / f"seed-{seed}.npy", state, allow_pickle=False)
            logger = api.get_trace_logger()
            logger.reset()
            namespace = api.functions()
            namespace.update(env=low, __name__="__main__")
            if args.phase == "snapshot":
                observation = low.get_observation()
                for camera, view in observation.items():
                    if isinstance(view, dict) and "images" in view and "rgb" in view["images"]:
                        Image.fromarray(view["images"]["rgb"]).save(trial_dir / f"{camera}.png")
                docs = [f"Task: {low.handle.task_language}", "", "Fixed API reference (signatures and docstrings):"]
                for name, fn in api.functions().items():
                    docs.extend(["", f"{name}{inspect.signature(fn)}", inspect.getdoc(fn) or ""])
                (trial_dir / "api-reference.txt").write_text("\n".join(docs) + "\n")
                continue
            low.enable_video_capture(True, clear=True)
            out, err = io.StringIO(), io.StringIO()
            started = time.monotonic()
            error = None
            try:
                signal.alarm(1000)
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    exec(compile(text, str(program), "exec"), namespace)
            except BaseException:
                error = traceback.format_exc()
            finally:
                signal.alarm(0)
            row = {"task_id": args.task_id, "seed": seed, "phase": args.phase, "version": args.version, "program_sha256": code_hash, "task_completed": bool(low.task_completed()), "simulator_steps": low._sim_step_count, "elapsed_seconds": time.monotonic() - started, "program_ok": error is None, "initial_state_sha256": digest, "trial_dir": str(trial_dir)}
            if args.phase == "evaluation":
                other = np.load(reference / "host-audit/evaluation" / f"task_{args.task_id}" / f"seed_{seed}" / "initial-state-0.npy", allow_pickle=False)
                row["matches_aspire_initial_state"] = bool(np.array_equal(state, other))
            (trial_dir / "stdout.txt").write_text(out.getvalue())
            (trial_dir / "stderr.txt").write_text(err.getvalue() + (error or ""))
            logger.save(trial_dir)
            frames = low.get_video_frames(clear=True)
            if frames:
                import imageio.v2 as imageio
                imageio.mimwrite(str(trial_dir / "video.mp4"), frames, fps=30, codec="libx264")
                (trial_dir / "keyframes").mkdir(exist_ok=True)
                for rank in range(min(10, len(frames))):
                    index = rank * (len(frames) - 1) // max(min(10, len(frames)) - 1, 1)
                    Image.fromarray(frames[index]).save(trial_dir / "keyframes" / f"video-{rank:02d}.jpg")
            (trial_dir / "result.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        (output / "complete.json").write_text(json.dumps({"finished_at": now(), "trials": len(rows), "successes": sum(row["task_completed"] for row in rows), "program_sha256": code_hash}, indent=2) + "\n")
    finally:
        low.handle.env.close()


if __name__ == "__main__":
    main()
