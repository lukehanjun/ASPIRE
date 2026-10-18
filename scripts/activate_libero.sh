# Source this file from any directory: source /path/to/ASPIRE/scripts/activate_libero.sh
_aspire_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_aspire_root/.venv-libero/bin/activate"
export PATH="$_aspire_root/artifacts/bin:$PATH"
export LIBERO_CONFIG_PATH="$_aspire_root/.libero"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
export PYGLET_HEADLESS=true
export CUDA_VISIBLE_DEVICES="${ASPIRE_SIM_GPU:-0}"
export MUJOCO_EGL_DEVICE_ID="${ASPIRE_SIM_GPU:-0}"
export CAP_HARNESS_PROFILE=a40
export CAP_HARNESS_VENDOR_ROOT="$_aspire_root/third_party"
export CAP_HARNESS_SAM3_URL=http://127.0.0.1:8214
export CAP_HARNESS_GRASPNET_URL=http://127.0.0.1:8215
export CAP_HARNESS_PYROKI_URL=http://127.0.0.1:8216
export CAP_HARNESS_CUROBO_URL=http://127.0.0.1:8218
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS="${MAX_JOBS:-4}"
export UV_CACHE_DIR="$_aspire_root/.uv-cache"
export CAP_HARNESS_PROVIDER_CACHE="$UV_CACHE_DIR"
export CAP_HARNESS_CUROBO_CACHE="$UV_CACHE_DIR"
export TMPDIR="$_aspire_root/artifacts/tmp"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="$_aspire_root/artifacts/cache"
export HF_HUB_CACHE="$_aspire_root/artifacts/cache/huggingface/hub"
export HF_XET_CACHE="$_aspire_root/artifacts/cache/huggingface/xet"
export TORCH_HOME="$_aspire_root/artifacts/cache/torch"
export TORCH_EXTENSIONS_DIR="$_aspire_root/artifacts/cache/torch-extensions"
export ROBOT_DESCRIPTIONS_CACHE="$_aspire_root/artifacts/cache/robot-descriptions"
export LD_LIBRARY_PATH="$_aspire_root/artifacts/system-libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"
unset _aspire_root
