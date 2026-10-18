"""Expose the released ASPIRE v1 API to the unmodified CaP-X trial loop.

The host-only initialization shim reproduces the reference baseline's scenes.
No task-specific programs, development episodes or learned libraries are used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

from capx.envs.base import BaseEnv
from capx.envs.tasks.base import CodeExecutionEnvBase
import numpy as np

from cap_harness import contracts
from cap_harness.api import CapApi
from cap_harness.artifacts import RunRecorder, TracedProvider
from cap_harness.libero.adapter import LiberoAdapter
from cap_harness.libero.registry import LiberoSuiteRegistry
from cap_harness.providers.curobo.client import CuRoboProvider
from cap_harness.providers.graspnet.client import ContactGraspNetProvider
from cap_harness.providers.pyroki.client import PyRokiProvider
from cap_harness.providers.sam3.client import Sam3Provider
from cap_harness.registry import ToolRegistry
from cap_harness.runtime import _SAFE_BUILTINS, _SAFE_CONSTRUCTORS, ProgramExecutor


def compact(value, depth=0):
    if depth > 5:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim <= 1 and value.size <= 32:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): compact(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v, depth + 1) for v in value[:32]]
    if is_dataclass(value):
        return {f.name: compact(getattr(value, f.name), depth + 1) for f in fields(value)}
    return str(value)


class MatchedLiberoAdapter(LiberoAdapter):
    """Use v1 control/perception with the v0.1 baseline's initialization only."""

    def reset(self, task_ref, seed):
        metadata = self.registry.resolve(task_ref)
        states = self.registry.get_init_states(metadata)
        env = self._create_environment(metadata)
        original = env.reset

        def legacy_reset():
            obs = original()
            if len(states):
                env.set_init_state(states[0])
                env.set_init_state(states[(seed - 1) % len(states)])
                obs = original()
            return obs

        env.reset = legacy_reset
        try:
            return super().reset(metadata, seed)
        finally:
            env.reset = original

    def _settle(self, env):
        for _ in range(self.post_settle_steps):
            action = np.zeros(8, dtype=np.float64)
            action[-1] = -1.0  # Original baseline settles with the gripper opening.
            (
                self._raw_observation,
                self._last_reward,
                self._last_terminated,
                self._last_truncated,
                self._last_native_info,
            ) = self._unpack_native_result(env.step(action))


class _VideoObserver:
    def __init__(self, low, recorder):
        self.low, self.recorder = low, recorder

    def __getattr__(self, name):
        return getattr(self.recorder, name)

    def after_step(self, action, result):
        self.recorder.after_step(action, result)
        if self.low._record_frames and self.recorder.step_count % 4 == 0 and result.observation:
            self.low._frames.append(result.observation.cameras["agentview"].rgb.copy())


class AspireV1Low(BaseEnv):
    """Compatibility surface for the CaP-X host; hidden from generated programs."""

    def __init__(
        self,
        suite_name,
        task_id,
        privileged=False,
        max_steps=8000,
        seed=None,
        enable_render=True,
        viser_debug=False,
        control_freq=20,
    ):
        super().__init__()
        self.suite_name, self.task_id = suite_name, task_id
        self.max_steps, self.privileged = max_steps, False
        self.registry = LiberoSuiteRegistry()
        self.metadata = self.registry.resolve(suite_name, task_id)
        self.adapter = MatchedLiberoAdapter(
            registry=self.registry,
            horizon=max_steps,
            camera_width=800,
            camera_height=512,
            control_frequency=control_freq,
            init_mode="seeded",
        )
        self.handle = SimpleNamespace(
            task_id=task_id, task_language=self.metadata.language, env=None
        )
        self._frames = []
        self._record_frames = False
        self.recorder = None
        self._phase = "setup"
        self.reset(seed=0)

    @property
    def _sim_step_count(self):
        return self.adapter._elapsed_steps + 10

    def reset(self, *, seed=None, options=None):
        observation = self.adapter.reset(self.metadata, int(seed or 0))
        self.handle.env = self.adapter.native_env
        self._frames.clear()
        return {"observation": observation}, {"task_prompt": self.metadata.language}

    def get_observation(self):
        return {"observation": self.adapter.get_observation()}

    def task_completed(self):
        return self.adapter.check_success()

    def compute_reward(self):
        return float(self.task_completed())

    def step(self, action):
        raise TypeError("generated code must use ASPIRE's public API")

    def render(self, mode="rgb_array"):
        return self.adapter.get_observation().cameras["agentview"].rgb

    def render_wrist(self):
        return self.adapter.get_observation().cameras["robot0_eye_in_hand"].rgb

    def enable_video_capture(self, enabled=True, *, clear=True, wrist_camera=False):
        self._record_frames = enabled
        if clear:
            self._frames.clear()

    def get_video_frames(self, *, clear=False):
        frames = list(self._frames)
        if clear:
            self._frames.clear()
        return frames

    def get_video_frame_count(self):
        return len(self._frames)

    def get_video_frames_range(self, start, end):
        return self._frames[start:end]

    def close(self):
        self.adapter.close()


class AspireV1CodeEnv(CodeExecutionEnvBase):
    """Run each code block using v1's restricted executor and fixed public API."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.low_level_env = self._build_low_level(
            cfg.low_level, cfg.privileged, cfg.enable_render, cfg.viser_debug
        )
        self._apis = {}
        self._step_count = 0
        self._task_prompt = cfg.prompt or "Goal: {libero_environment_goal}"
        self._full_prompt = []
        self._phase = "setup"
        self._attempt = 0
        self._execution = None

    def reset(self, *, seed=None, options=None):
        low = self.low_level_env
        low.adapter._run_observer = None
        obs, info = low.reset(seed=seed, options=options)
        root = Path(os.environ["ASPIRE_V1_CAP_AGENT_RUN_ROOT"])
        self.recorder = RunRecorder(
            output_root=root
            / "v1-artifacts"
            / self._phase
            / f"task_{low.task_id}"
            / f"attempt_{self._attempt}",
            benchmark="libero-pro",
            suite=low.suite_name,
            task_id=low.task_id,
            task_name=low.metadata.task_name,
            seed=int(seed),
            max_steps=low.max_steps - 10,
            camera_width=800,
            camera_height=512,
            init_mode="baseline-paired-host-shim",
            flat_layout=True,
            capture_videos=False,
        )
        self._attempt += 1
        self._step_count = 0
        low.recorder = self.recorder
        low.adapter._run_observer = _VideoObserver(low, self.recorder)
        self.recorder.on_reset(low.adapter.get_observation(), low.metadata.to_manifest_record())
        sam = TracedProvider(Sam3Provider(base_url="http://127.0.0.1:8214"), self.recorder, "sam3")
        grasp = TracedProvider(
            ContactGraspNetProvider(base_url="http://127.0.0.1:8215"),
            self.recorder,
            "contact_graspnet",
        )
        ik = TracedProvider(
            PyRokiProvider(base_url="http://127.0.0.1:8216"), self.recorder, "pyroki"
        )
        curobo = TracedProvider(
            CuRoboProvider(base_url="http://127.0.0.1:8218"), self.recorder, "curobo"
        )
        self.api = CapApi(
            low.adapter,
            segmentation_provider=sam,
            grasp_providers={"contact-graspnet": grasp},
            ik_providers={"pyroki": ik, "curobo": curobo},
            trajectory_planning_providers={"curobo": curobo},
            integrated_pose_planning_providers={"curobo-integrated": curobo},
            default_camera="agentview",
            recorder=self.recorder,
        )
        registry = self.api.register_tools(ToolRegistry())
        self.executor = ProgramExecutor(registry, recorder=self.recorder)
        docs = []
        for name, constructor in _SAFE_CONSTRUCTORS.items():
            docs.append(f"{name}{inspect.signature(constructor)}")
        for name, value in vars(contracts).items():
            if (
                inspect.isclass(value)
                and is_dataclass(value)
                and value.__module__ == contracts.__name__
            ):
                docs.append(
                    f"Returned contract {name}: "
                    + "; ".join(f"{f.name}: {f.type}" for f in fields(value))
                )
        for spec in registry.all_specs(public_only=True):
            doc = inspect.getdoc(spec.function) or ""
            docs.append(f"{spec.name}{spec.signature}\n{doc}")
        instructions = (
            "You control a Franka robot through ASPIRE's fixed public API. Goal: {libero_environment_goal}\n"
            "Write executable Python only. The scene persists between code blocks; each block has fresh Python locals. "
            "Only import math; all public functions and constructors below are injected. NumPy imports are forbidden. "
            "Use geometry returned by perception and measured robot state; no simulator internals or filesystem/network access. "
            "Pose positions are in metres, quaternion_wxyz is [w,x,y,z], and robot_base is the robot base frame. "
            "MotionStrategy() uses PyRoki and joint interpolation by default; cuRobo planning is available explicitly. "
            "Print concise numeric observations/results for feedback, or set result to a public result object. "
            "After execution, stdout/errors and visual-difference descriptions let you write another code block.\n"
        )
        instructions += (
            "Runtime rules: variable and function names must not begin with underscore; private attributes are forbidden. "
            "Classes, with, from-imports, async, global, nonlocal and delete are forbidden. "
            "The available Python built-ins are exactly: "
            + ", ".join(name for name in _SAFE_BUILTINS if not name.startswith("_"))
            + ". Other built-ins, including hasattr, getattr, isinstance, type and AttributeError, are unavailable. "
            "For this single-arm embodiment, get_robot_state().joint_positions, joint_velocities, "
            "end_effector_poses and gripper_positions are mappings keyed by 'primary'. "
            "Read the pose with get_robot_state().end_effector_poses['primary']. "
            "For perception, localize_object(query).geometry.pose is the estimated object pose; "
            "check .ok before accessing optional results. Arrays support public indexing, .tolist() and arithmetic.\n"
        )
        docs.append((root / "source/aspire-v1/docs/api-reference.md").read_text())
        instructions = instructions.replace("{libero_environment_goal}", low.metadata.language)
        self._full_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instructions + "\nPublic API:\n" + "\n\n".join(docs)}
                ],
            }
        ]
        self._execution = None
        obs.update(full_prompt=self._full_prompt)
        return obs, info

    def step(self, action):
        self._step_count += 1
        path = self.recorder.root / "source" / f"block_{self._step_count:02d}.py"
        path.write_text(action)
        execution = self.executor.execute_program(action)
        self._execution = execution
        task_completed = bool(self.low_level_env.task_completed())
        stdout = execution.stdout
        if execution.result is not None:
            stdout += "\nPublic program result: " + json.dumps(compact(execution.result))
        stderr = "" if execution.ok else json.dumps(compact(execution.error))
        obs = self.low_level_env.get_observation()
        obs["full_prompt"] = self._full_prompt
        truncated = (
            self.low_level_env._sim_step_count >= self.low_level_env.max_steps
            or self.recorder.terminal_reason == "step_limit"
        )
        return (
            obs,
            float(task_completed),
            task_completed,
            truncated,
            {
                "sandbox_rc": 0 if execution.ok else 1,
                "stdout": stdout,
                "stderr": stderr,
                "task_completed": task_completed,
                "task_prompt": self._task_prompt,
            },
        )

    def finalize(self, summary):
        self.recorder.finalize(
            program_ok=bool(summary.success),
            task_success=bool(summary.task_completed),
            termination_reason="task_succeeded"
            if summary.task_completed
            else "step_limit"
            if summary.truncated
            else "program_completed",
            program_result=self._execution,
        )

    def _get_observation(self):
        return {**self.low_level_env.get_observation(), "full_prompt": self._full_prompt}

    def close(self):
        self.low_level_env.close()
