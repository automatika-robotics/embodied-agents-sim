#!/usr/bin/env bash
# Installer for the embodied-agents sim stack. Safe to re-run: completed
# phases are recorded in .setup-state/ and skipped, and downloads resume.
#
#   ./doctor.sh          # check prerequisites first
#   ./setup.sh           # install everything
#   ./setup.sh --verify  # only run the final checks
#
# Installs under SIM_ROOT, by default this repo's parent directory:
# lerobot-venv/  isaaclab-venv/  IsaacLab/  leisaac/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.env"
SIM_ROOT="${SIM_ROOT:-$(dirname "$HERE")}"
STATE="$HERE/.setup-state"
mkdir -p "$STATE"

export HF_HUB_DOWNLOAD_TIMEOUT=60
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=4
export OMNI_KIT_ACCEPT_EULA=YES
PIP_NET=(--retries 10 --timeout 60 --resume-retries 999)

log()  { echo -e "\n=== $* ==="; }
stamp(){ touch "$STATE/$1.done"; }
done_p(){ [ -f "$STATE/$1.done" ]; }

retry() { # retry <n> <cmd...>, 10 s between attempts
    local n=$1; shift
    local i=0
    until "$@"; do
        i=$((i+1))
        [ "$i" -ge "$n" ] && { echo "FAILED after $n attempts: $*"; return 1; }
        echo "retrying ($i/$n): $*"; sleep 10
    done
}

find_python() { # find_python <ver>: system python, else one installed by uv
    local ver=$1
    if command -v "python$ver" >/dev/null; then command -v "python$ver"; return; fi
    if command -v uv >/dev/null; then
        uv python install "$ver" >&2
        uv python find "$ver"
        return
    fi
    echo "python$ver not found and uv unavailable: install one of them" >&2
    return 1
}

verify_all() {
    log "verify: lerobot venv"
    "$SIM_ROOT/lerobot-venv/bin/python" - <<'EOF'
import grpc, lerobot
print(f"  lerobot {lerobot.__version__} + grpcio OK")
EOF
    "$SIM_ROOT/lerobot-venv/bin/python" "$HERE/patches/apply_lerobot_patches.py" --check | sed 's/^/  /'
    log "verify: isaac venv"
    "$SIM_ROOT/isaaclab-venv/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA device"
print(f"  torch {torch.__version__}, cuda OK")
from isaaclab.app import AppLauncher  # noqa
print("  isaaclab AppLauncher OK")
import leisaac  # noqa
print("  leisaac OK")
EOF
    log "verify: scene assets"
    test -f "$SIM_ROOT/leisaac/assets/scenes/kitchen_with_orange/scene.usd" && echo "  kitchen scene OK"
    test -f "$SIM_ROOT/leisaac/assets/robots/so101_follower.usd" && echo "  robot USD OK"
    echo -e "\nAll verifications passed."
}

if [ "${1:-}" = "--verify" ]; then verify_all; exit 0; fi

# ---------------------------------------------------------------- phase 0
if ! done_p 00-gitlfs; then
    log "phase 0: git-lfs (needed by the IsaacLab clone)"
    if ! command -v git-lfs >/dev/null; then
        mkdir -p "$HOME/.local/bin"
        tmp=$(mktemp -d)
        retry 20 curl -sL --retry 10 --retry-delay 5 -C - -o "$tmp/lfs.tgz" "$GIT_LFS_URL"
        tar xzf "$tmp/lfs.tgz" -C "$tmp"
        cp "$tmp"/git-lfs-*/git-lfs "$HOME/.local/bin/"
        rm -rf "$tmp"
        export PATH="$HOME/.local/bin:$PATH"
    fi
    git lfs install
    stamp 00-gitlfs
fi

# ---------------------------------------------------------------- phase 1
if ! done_p 10-lerobot-venv; then
    log "phase 1: LeRobot policy-server venv (python $PY_LEROBOT)"
    PY=$(find_python "$PY_LEROBOT")
    [ -d "$SIM_ROOT/lerobot-venv" ] || "$PY" -m venv "$SIM_ROOT/lerobot-venv"
    "$SIM_ROOT/lerobot-venv/bin/pip" install -q --upgrade pip
    retry 30 "$SIM_ROOT/lerobot-venv/bin/pip" install "$LEROBOT_PIP_SPEC" "${PIP_NET[@]}"
    stamp 10-lerobot-venv
fi

if ! done_p 11-lerobot-patches; then
    log "phase 1b: lerobot server patches"
    "$SIM_ROOT/lerobot-venv/bin/python" "$HERE/patches/apply_lerobot_patches.py"
    stamp 11-lerobot-patches
fi

# ---------------------------------------------------------------- phase 2
if ! done_p 20-isaac-venv; then
    log "phase 2: Isaac Sim venv (python $PY_ISAAC), the big download"
    PY=$(find_python "$PY_ISAAC")
    [ -d "$SIM_ROOT/isaaclab-venv" ] || "$PY" -m venv "$SIM_ROOT/isaaclab-venv"
    "$SIM_ROOT/isaaclab-venv/bin/pip" install -q --upgrade pip
    retry 60 "$SIM_ROOT/isaaclab-venv/bin/pip" install $ISAACSIM_PIP_SPEC \
        --extra-index-url "$ISAACSIM_INDEX_URL" "${PIP_NET[@]}"
    stamp 20-isaac-venv
fi

if ! done_p 21-torch-pin; then
    log "phase 2b: torch $TORCH_PIP_SPEC"
    retry 60 "$SIM_ROOT/isaaclab-venv/bin/pip" install $TORCH_PIP_SPEC \
        --index-url "$TORCH_INDEX_URL" "${PIP_NET[@]}"
    stamp 21-torch-pin
fi

if ! done_p 22-isaaclab; then
    log "phase 2c: IsaacLab $ISAACLAB_TAG"
    if [ ! -d "$SIM_ROOT/IsaacLab/.git" ]; then
        retry 20 git clone --branch "$ISAACLAB_TAG" --depth 1 "$ISAACLAB_REPO" "$SIM_ROOT/IsaacLab"
    else
        git -C "$SIM_ROOT/IsaacLab" fetch --depth 1 origin tag "$ISAACLAB_TAG" || true
        git -C "$SIM_ROOT/IsaacLab" checkout "$ISAACLAB_TAG"
    fi
    # These two old sdists fail to build under modern setuptools and cmake 4,
    # so they are built here with the venv's own toolchain
    retry 10 env CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        "$SIM_ROOT/isaaclab-venv/bin/pip" install --no-build-isolation flatdict==4.0.1 egl_probe "${PIP_NET[@]}"
    stamp 22-isaaclab
fi

if ! done_p 23-isaaclab-install; then
    log "phase 2d: IsaacLab --install"
    ( cd "$SIM_ROOT/IsaacLab" && \
      retry 20 env VIRTUAL_ENV="$SIM_ROOT/isaaclab-venv" PATH="$SIM_ROOT/isaaclab-venv/bin:$PATH" \
        ./isaaclab.sh --install )
    stamp 23-isaaclab-install
fi

# ---------------------------------------------------------------- phase 3
if ! done_p 30-leisaac; then
    log "phase 3: leisaac @ ${LEISAAC_COMMIT:0:9}"
    if [ ! -d "$SIM_ROOT/leisaac/.git" ]; then
        retry 20 git clone "$LEISAAC_REPO" "$SIM_ROOT/leisaac"
    fi
    git -C "$SIM_ROOT/leisaac" checkout "$LEISAAC_COMMIT"
    ( cd "$SIM_ROOT/leisaac" && \
      retry 20 "$SIM_ROOT/isaaclab-venv/bin/pip" install -e "source/leisaac[lerobot-async]" "${PIP_NET[@]}" )
    stamp 30-leisaac
fi

if ! done_p 31-leisaac-patches; then
    log "phase 3b: leisaac native-client compat patches (for lerobot >= 0.6 servers)"
    python3 "$HERE/patches/apply_leisaac_patches.py" --leisaac "$SIM_ROOT/leisaac"
    stamp 31-leisaac-patches
fi

if ! done_p 32-assets; then
    log "phase 3c: scene assets"
    mkdir -p "$SIM_ROOT/leisaac/assets/scenes" "$SIM_ROOT/leisaac/assets/robots"
    retry 20 curl -L --retry 10 --retry-delay 5 -C - \
        -o "$SIM_ROOT/leisaac/assets/kitchen_with_orange.zip" "$ASSETS_SCENE_URL"
    retry 20 curl -L --retry 10 --retry-delay 5 -C - \
        -o "$SIM_ROOT/leisaac/assets/robots/so101_follower.usd" "$ASSETS_ROBOT_URL"
    unzip -q -o "$SIM_ROOT/leisaac/assets/kitchen_with_orange.zip" -d "$SIM_ROOT/leisaac/assets/scenes/"
    rm -f "$SIM_ROOT/leisaac/assets/kitchen_with_orange.zip"
    stamp 32-assets
fi

# ---------------------------------------------------------------- done
verify_all
cat <<EOF

Setup complete. Everything lives under: $SIM_ROOT

Next steps (see README):
  ./doctor.sh                 re-check the environment any time
  make server                 terminal A: LeRobot policy server (port 8080)
  make bridge                 terminal B: Isaac Sim + ROS 2 bridge
  make status                 check topics, rates and VRAM
  make smoke                  optional: end-to-end policy-server smoke test
Then run a recipe and start the task (see "Running an evaluation" in the README).

The first Isaac launch compiles shaders and can take 10-20 minutes with an
unresponsive window. Later launches take 1-2 minutes.
EOF
