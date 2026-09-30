#!/usr/bin/env bash
# Start Isaac Sim and the LeIsaac ROS 2 bridge in the foreground.
#
# Runs from the leisaac checkout, since scene assets resolve relative to it.
# If the system rclpy targets another Python than Isaac's 3.11 venv (Jazzy is
# 3.12), falls back to the ROS 2 libraries bundled with Isaac Sim.
#
# Usage:  [HEADLESS=1] scripts/run_bridge.sh [extra bridge args]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_ROOT="${SIM_ROOT:-$(dirname "$HERE")}"
PYBIN="$SIM_ROOT/isaaclab-venv/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES

ROS_DISTRO_DIR=$(ls /opt/ros 2>/dev/null | head -1)
[ -n "$ROS_DISTRO_DIR" ] || { echo "ERROR: no ROS 2 install in /opt/ros"; exit 1; }

# ROS setup.bash reads unset variables, so nounset is off around it
set +u
source "/opt/ros/$ROS_DISTRO_DIR/setup.bash" 2>/dev/null || true
set -u
if "$PYBIN" -c "import rclpy" 2>/dev/null; then
    echo "Using system ROS 2 ($ROS_DISTRO_DIR) rclpy"
else
    EXT="$SIM_ROOT/isaaclab-venv/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/$ROS_DISTRO_DIR"
    if [ ! -d "$EXT/rclpy" ]; then
        echo "ERROR: system rclpy not importable from Isaac's python and no internal libs for '$ROS_DISTRO_DIR'"
        exit 1
    fi
    export PYTHONPATH="$EXT/rclpy${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$EXT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Using Isaac Sim's bundled ROS 2 ($ROS_DISTRO_DIR, cp311) client libraries"
fi

ARGS=(--task=LeIsaac-SO101-PickOrange-v0 --enable_cameras)
[ "${HEADLESS:-0}" = "1" ] && ARGS+=(--headless)

echo "Launching bridge (first Isaac launch compiles shaders: 10-20 min; later launches ~1-2 min)"
cd "$SIM_ROOT/leisaac"
exec "$PYBIN" -u "$HERE/scripts/leisaac_ros2_bridge.py" "${ARGS[@]}" "$@"
