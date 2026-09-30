#!/usr/bin/env bash
# Preflight checks for the embodied-agents sim stack. Changes nothing; prints
# PASS, WARN or FAIL with a fix for each problem, and exits non-zero on a FAIL.
#
#   ./doctor.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/versions.env"
SIM_ROOT="${SIM_ROOT:-$(dirname "$HERE")}"

pass=0; warn=0; fail=0
ok()   { echo "  [PASS] $1"; pass=$((pass+1)); }
wrn()  { echo "  [WARN] $1"; [ -n "${2:-}" ] && echo "         fix: $2"; warn=$((warn+1)); }
bad()  { echo "  [FAIL] $1"; [ -n "${2:-}" ] && echo "         fix: $2"; fail=$((fail+1)); }

echo "== GPU =="
if command -v nvidia-smi >/dev/null; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    major=${drv%%.*}
    if [ "${major:-0}" -ge 535 ]; then ok "driver $drv (>= 535)"; else bad "driver $drv < 535" "upgrade the NVIDIA driver"; fi
    if [ "${vram:-0}" -ge 22000 ]; then
        ok "VRAM ${vram} MiB (GR00T server ~9.5 GiB + Isaac ~6 GiB fit)"
    else
        wrn "VRAM ${vram} MiB is too little for the GR00T server and Isaac together" "serve the policy on another machine, or use the SmolVLA fallback checkpoint"
    fi
else
    bad "nvidia-smi not found" "install the NVIDIA driver"
fi

echo "== Disk =="
avail_gb=$(df --output=avail -BG "$SIM_ROOT" | tail -1 | tr -dc '0-9')
if [ "${avail_gb:-0}" -ge 80 ]; then ok "${avail_gb}G free at $SIM_ROOT"; else
    if [ -d "$SIM_ROOT/isaaclab-venv" ]; then wrn "${avail_gb}G free (installs already present, may be enough)"; else
        bad "${avail_gb}G free < 80G needed for a fresh install" "free disk space or set SIM_ROOT elsewhere"; fi
fi

echo "== Python =="
for spec in "ISAAC:$PY_ISAAC" "LEROBOT:$PY_LEROBOT"; do
    name=${spec%%:*}; ver=${spec##*:}
    if command -v "python$ver" >/dev/null; then ok "python$ver present ($name venv)"
    elif command -v uv >/dev/null; then wrn "python$ver missing but uv can install it" "setup.sh will run: uv python install $ver"
    else bad "python$ver missing and no uv" "install uv (https://astral.sh/uv) or python$ver; no sudo needed for uv"; fi
done

echo "== ROS 2 =="
distro=$(ls /opt/ros 2>/dev/null | head -1)
if [ -n "$distro" ]; then
    ok "ROS 2 $distro found"
    case "$distro" in humble|jazzy) : ;; *) wrn "distro '$distro' untested (validated: jazzy, humble)";; esac
else
    bad "no ROS 2 in /opt/ros" "sudo apt install ros-jazzy-ros-base python3-colcon-common-extensions (24.04) or ros-humble-ros-base (22.04)"
fi

echo "== git-lfs =="
if command -v git-lfs >/dev/null; then ok "git-lfs $(git-lfs version 2>/dev/null | cut -d' ' -f1 | cut -d/ -f2)"; else
    wrn "git-lfs missing (IsaacLab clone needs it)" "setup.sh installs a standalone binary to ~/.local/bin"; fi

echo "== HuggingFace =="
if hfout=$(hf auth whoami 2>/dev/null) || hfout=$(huggingface-cli whoami 2>/dev/null); then
    ok "logged in: $(echo "$hfout" | tail -1 | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.*: *//')"
    # Access to the gated model every GR00T checkpoint loads
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
        -H "Authorization: Bearer $(cat ~/.cache/huggingface/token 2>/dev/null)" \
        "https://huggingface.co/nvidia/Cosmos-Reason2-2B/resolve/main/config.json" 2>/dev/null)
    case "$code" in
        200|302) ok "access to gated nvidia/Cosmos-Reason2-2B" ;;
        401|403) bad "no access to gated nvidia/Cosmos-Reason2-2B (every GR00T checkpoint loads it)" "request access on its HF page, then retry" ;;
        *)       wrn "could not verify gated-model access (HTTP $code, network?)" ;;
    esac
else
    bad "not logged in to HuggingFace" "run: hf auth login"
fi
if [ -d "$HOME/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B" ]; then
    ok "Cosmos-Reason2-2B already in local HF cache (no download needed)"
fi

echo "== Port hygiene =="
n8080=$(ss -tln 2>/dev/null | grep -c ':8080 ' || true)
if [ "${n8080:-0}" -eq 0 ]; then ok "port 8080 free"
elif [ "$n8080" -eq 1 ]; then wrn "one listener already on 8080" "if it is a stale policy server, kill it before starting a new one"
else bad "$n8080 listeners on port 8080: gRPC lets servers share the port and splits requests between them" "kill all: pkill -9 -f lerobot.async_inference.policy_server"; fi

echo "== Native-install shadowing =="
# A native EMOS install puts these packages in the ROS distro's site-packages,
# where they shadow a development workspace in processes launched without its
# PYTHONPATH, such as the UI node
distro_dir=$(ls -d /opt/ros/*/lib/python3*/site-packages 2>/dev/null | head -1)
if [ -n "$distro_dir" ]; then
    shadows=$(ls -d "$distro_dir"/{ros_sugar,agents,automatika_*,kompass*} 2>/dev/null | tr '\n' ' ')
    if [ -n "${shadows// /}" ]; then
        wrn "native emos packages found in $distro_dir: $shadows" \
            "if developing against a workspace (~/ros_ws), remove them or workspace changes will be silently shadowed in spawned processes"
    else
        ok "no automatika packages shadowing the workspace in the ROS distro dir"
    fi
fi

echo "== Zombie processes =="
z=$(ps -eo pid,comm | awk '$2=="Launcher"{print $1}' | head -5 | tr '\n' ' ')
if [ -n "${z// /}" ]; then
    wrn "sugarcoat 'Launcher' process(es) running: $z" \
        "fine if you are running a recipe; a leftover one keeps its node names (such as /vla_sim) and takes goals meant for new runs: make clean-procs"
else
    ok "no recipe Launcher processes"
fi

echo "== CPU governor =="
# The bridge is CPU-bound: in powersave it steps too slowly for the policy's
# timing (see "Timing configuration" in the README)
govs=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort -u | xargs)
if [ -z "${govs// /}" ]; then wrn "could not read the CPU frequency governor"
elif [ "${govs// /}" = "performance" ]; then ok "CPU governor: performance"
else wrn "CPU governor: $govs" "sudo cpupower frequency-set -g performance"; fi

echo "== Display =="
if [ -n "${DISPLAY:-}" ] || ls /tmp/.X11-unix/X* >/dev/null 2>&1; then ok "X display available (Isaac GUI + episode reset)"; else
    wrn "no X display detected" "run Isaac headless (--headless) and reset episodes via the /so101/reset service"; fi

echo
echo "Summary: $pass passed, $warn warnings, $fail failures"
[ "$fail" -eq 0 ] || exit 1
