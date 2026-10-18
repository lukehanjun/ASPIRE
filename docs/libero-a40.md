# LIBERO on this A40 host

The checkout is `/workspace/hanjun/ASPIRE`. Use Python 3.11 and the pinned
LIBERO-Pro and robosuite submodules. Environments, downloads, temporary files,
and compiled kernel caches live on the workspace disk.

```bash
cd /workspace/hanjun/ASPIRE
source scripts/activate_libero.sh
```

This activates `.venv-libero`, configures headless EGL rendering, and selects
the `a40` hardware profile. Simulation uses physical GPU 0; set
`ASPIRE_SIM_GPU=3` before sourcing to use a different free device. SAM3 uses
GPU 4, cuRobo uses GPU 5, and Contact-GraspNet uses GPU 6. PyRoki runs on CPU. GPUs 1 and 2 are occupied by
other projects. Check `nvidia-smi` before changing placement.

The profile uses private localhost ports: SAM3 8214, PyRoki 8216, cuRobo 8218,
and Contact-GraspNet 8215. The activation script exports matching
client URLs so episodes use this checkout's services.

Start the providers in a separate terminal:

```bash
scripts/start_libero_services.sh
```

Or keep them running after disconnecting:

```bash
mkdir -p validation-artifacts/service-logs
setsid nohup scripts/start_libero_services.sh \
  > validation-artifacts/service-logs/supervisor.log 2>&1 < /dev/null &
echo $! > validation-artifacts/service-logs/supervisor.pid
```

Stop that supervisor and its children with:

```bash
kill -TERM "$(cat validation-artifacts/service-logs/supervisor.pid)"
```

Run diagnostics and an example:

```bash
cap-harness doctor --runtime libero --expected-environment-name .venv-libero \
  --providers sam3,contact_graspnet,pyroki,curobo --reset \
  --output validation-artifacts/libero/doctor.json

cap-harness run --benchmark libero-pro --suite libero_goal_swap --task-id 0 \
  --seed 1 --program examples/libero/open_middle_drawer.py --output-root outputs
```

To rebuild the environments, keep the same activation settings and run:

```bash
scripts/bootstrap_libero.sh
scripts/bootstrap_providers.sh --providers sam3,contact_graspnet,pyroki,curobo --profile a40
```

The `uv` executable is in `artifacts/bin`. The host is missing the GLVND EGL
dispatch library, so Ubuntu's `libegl1` package was extracted into
`artifacts/system-libs`; the activation script adds it to `LD_LIBRARY_PATH`.
The ignored robosuite `macros_private.py` disables its shared `/tmp/robosuite.log`
file; console messages are captured by the per-run and diagnostic logs.
Hugging Face credentials remain in the existing user credential store.
SAM3 weights are cached in `artifacts/cache/huggingface/hub`.

The cuRobo bootstrap pins Warp `1.18.0+cu12` and verifies CUDA initialization.
The default PyPI wheel uses CUDA 13; this driver's CUDA 12 environment needs the
[official compatibility wheel](https://nvidia.github.io/warp/latest/user_guide/installation.html#installing-from-github-releases).

Contact-GraspNet is installed in `.venv-graspnet` and starts with the helper.
`PYGLET_HEADLESS=true` prevents its renderer imports from requiring X11. During
the initial transition it runs under a second supervisor recorded in
`validation-artifacts/graspnet-service-logs/supervisor.pid`; stop that supervisor
as well before restarting the combined four-provider stack. The complete benchmark
workflow and seed partitions are documented in [libero-pro.md](libero-pro.md)
and the iterative-debugging skill.

## Released CaP-Agent0 comparison

The Haiku 5.5 baseline uses the v0.1 simulation sources at commit
`317a72e649d9c083298a1724d873491e499c0712`, extracted under
`artifacts/releases/aspire-v0.1`, with its frozen `.venv-cap-agent0` environment.
It uses the original fixed API library and image-difference feedback, generates
a fresh program for every episode, and preserves the released retry/reset
behavior. The batch runner overrides the YAML horizon to **8,000 steps**.
The ten `libero_goal_swap` tasks each use seeds 51–65, for 150 trials.
Haiku replaces both the original code-generation and visual-feedback models;
this is a new measurement, separate from the paper's ASPIRE 68% result.

Its original SAM3 and Contact-GraspNet servers use GPUs 4 and 6 on private ports
8234 and 8235; the original PyRoki server runs on CPU at port 8236. Contact-GraspNet
uses its v0.1 revision `2d71da4e` plus the release's verified three-file compatibility
patch. The API also auto-detects the existing Molmo2-8B service on GPU 2 at port
8122. The baseline does not use cuRobo. Simulation workers use GPUs 0, 3 and 7.

`scripts/cap_agent0_gateway.py` holds the Vector credential, records model calls,
and enforces the API budget on localhost 8230. `scripts/run_cap_agent0.py` wraps
the original runner to select Haiku, redirect private service ports, and record
native success and initial-state fingerprints. `scripts/run_cap_agent0_campaign.py`
supervises the GPU groups and writes `progress.md` and `summary.json`. The active
run directory is recorded in `artifacts/paper-audit/baseline-root.txt`; its
`baseline.json` contains the exact settings, versions, hashes, and exceptions.

The initial attempt that reused the newer unseeded IK service is marked invalid
and excluded. Use the original PyRoki service for this comparison. The newer
provider's zero-joint default was corrected in source and checked by forward
kinematics. Its live server was restarted with the fix and independently checked;
see `validation-artifacts/libero/pyroki-live-initialization-fix.json`. The baseline
continues using its separate original v0.1 IK server throughout.

## Verification on 2026-10-17

- Full diagnostics, including simulator reset and all three service checks:
  `validation-artifacts/libero/doctor.json` — all checks passed.
- Four family representatives at seed 1 (Object, Spatial, Goal, Long):
  `validation-artifacts/libero/setup-smoke/summary.json` — 4/4 passed reset,
  RGB-D/schema, hold/movement tick, and close checks. The manifest discovers all
  80 required pairs; this setup check runs four representatives.
- cuRobo GPU IK and collision-aware joint trajectory planning:
  `validation-artifacts/libero/curobo-smoke.json` — both passed. The first
  request compiles kernels and can take several minutes; the running service
  retains its initialized solvers.
- Focused Python regression tests: 46 passed; Ruff, shell syntax, and whitespace
  checks passed. Re-running the bootstraps skips unchanged environments.
- The supplied drawer example ran with SAM3 perception and PyRoki-driven robot
  motion, but timed out during its orientation move and did not complete the
  task. Its artifacts are under `outputs/setup-check/`; the console log is
  `artifacts/setup-logs/example.log`. No task-success benchmark is claimed.
