"""ROS 2 bridge for LeIsaac environments.

Runs a LeIsaac task in Isaac Sim and exposes it over ROS 2 for the
embodied-agents VLA component. The environment setup follows leisaac's
scripts/evaluation/policy_inference.py, with the policy client replaced by
ROS 2 topics. Joint values are converted between sim radians and the SO101
motor units of LeRobot datasets, so every topic carries motor units.

Start it with `make bridge`, which sets up the Isaac venv, rclpy and the
working directory. Validated against the leisaac commit in versions.env.
"""

import multiprocessing
from collections import deque

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="ROS2 bridge for LeIsaac environments.")
parser.add_argument(
    "--task", type=str, default="LeIsaac-SO101-PickOrange-v0", help="Name of the task."
)
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--camera_hz", type=int, default=15, help="Camera publishing rate in Hz."
)
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument(
    "--namespace", type=str, default="/so101", help="Namespace for the ROS2 topics."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Everything below must be imported after the app is launched
import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import torch  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import leisaac  # noqa: F401, E402  (registers the gym tasks)
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES  # noqa: E402
from leisaac.utils.env_utils import (  # noqa: E402
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
from leisaac.utils.robot_utils import (  # noqa: E402
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)


class ResetKeyListener:
    """Listen for the R key in the Isaac window to reset the episode."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        self.reset_requested = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_requested = True
        return True


class LeIsaacBridge(Node):
    """Publishes sim observations (motor units) and receives joint commands."""

    def __init__(self, namespace: str):
        super().__init__("leisaac_bridge")
        self.joint_state_pub = self.create_publisher(
            JointState, f"{namespace}/joint_states", 10
        )
        self.image_pubs = {}
        self.cmd_sub = self.create_subscription(
            JointState, f"{namespace}/joint_cmd", self._on_command, 10
        )
        self._namespace = namespace
        # Commands in motor units, one applied per env step as LeIsaac's own
        # evaluation client does. Keeping only the latest would skip or double
        # actions. Bounded so a runaway publisher cannot build up lag
        self.target_queue: deque = deque(maxlen=64)
        # Last command taken from the queue, and the target applied this step
        self.pending_cmd: np.ndarray | None = None
        self.applied_motor: np.ndarray | None = None
        # Episode reset without the GUI, same as the R key
        self.reset_requested = False
        self.reset_srv = self.create_service(
            Trigger, f"{namespace}/reset", self._on_reset
        )
        # "YES" whenever the env's success condition fires
        self.success_pub = self.create_publisher(
            String, f"{namespace}/task_success", 10
        )

    def _on_reset(self, request, response):
        self.reset_requested = True
        response.success = True
        response.message = "episode reset requested"
        return response

    def add_camera(self, key: str):
        self.image_pubs[key] = self.create_publisher(
            Image, f"{self._namespace}/{key}/image_raw", 10
        )

    def _on_command(self, msg: JointState):
        """Queue a joint command, in motor units, in the sim's joint order."""
        if msg.name:
            try:
                ordered = [
                    msg.position[list(msg.name).index(j)]
                    for j in SINGLE_ARM_JOINT_NAMES
                ]
            except ValueError:
                self.get_logger().warning(
                    f"Command joint names {list(msg.name)} do not cover "
                    f"{SINGLE_ARM_JOINT_NAMES}, ignoring command"
                )
                return
        else:
            if len(msg.position) != len(SINGLE_ARM_JOINT_NAMES):
                self.get_logger().warning("Unnamed command with wrong size, ignoring")
                return
            ordered = list(msg.position)

        self.target_queue.append(np.asarray(ordered, dtype=np.float64))

    def publish_joint_state(self, joint_pos: torch.Tensor, stamp):
        """Publish the sim joint positions, converted to motor units."""
        motor_pos = convert_leisaac_action_to_lerobot(joint_pos)
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(SINGLE_ARM_JOINT_NAMES)
        msg.position = [float(v) for v in motor_pos[0]]
        self.joint_state_pub.publish(msg)

    def publish_image(self, key: str, image: np.ndarray, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = key
        msg.height, msg.width = image.shape[0], image.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = image.shape[1] * 3
        msg.data = np.ascontiguousarray(image).tobytes()
        self.image_pubs[key].publish(msg)


def main():
    # Environment setup mirrors leisaac's policy_inference.py
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    rclpy.init()
    bridge = LeIsaacBridge(args_cli.namespace)
    camera_keys = [
        key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)
    ]
    for key in camera_keys:
        bridge.add_camera(key)
    bridge.get_logger().info(
        f"Bridging task '{args_cli.task}' with cameras {camera_keys} "
        f"on namespace '{args_cli.namespace}' (all joint values in motor units)"
    )

    try:
        listener = ResetKeyListener()
    except Exception:
        # Headless: no keyboard, reset through the service instead
        class _NoListener:
            reset_requested = False

        listener = _NoListener()
        bridge.get_logger().info(
            "No GUI keyboard available; reset episodes via the "
            f"'{args_cli.namespace}/reset' service"
        )

    obs_dict, _ = env.reset()
    step_dt = 1.0 / args_cli.step_hz
    camera_every = max(1, args_cli.step_hz // args_cli.camera_hz)
    step_count = 0

    while simulation_app.is_running():
        loop_start = time.perf_counter()
        with torch.inference_mode():
            if listener.reset_requested or bridge.reset_requested:
                listener.reset_requested = False
                bridge.reset_requested = False
                obs_dict, _ = env.reset()
                bridge.pending_cmd = None
                bridge.applied_motor = None
                bridge.target_queue.clear()

            policy_obs = obs_dict["policy"]
            joint_pos = policy_obs["joint_pos"]

            # One queued command per step; hold the last one when none is queued
            if bridge.target_queue:
                bridge.pending_cmd = bridge.target_queue.popleft()

            # Latch the hold pose once. Re-commanding the measured pose every
            # step lets gravity pull the arm down
            if bridge.applied_motor is None:
                bridge.applied_motor = convert_leisaac_action_to_lerobot(
                    joint_pos
                )[0].astype(np.float64)

            if bridge.pending_cmd is not None:
                bridge.applied_motor = bridge.pending_cmd.copy()

            sim_action = convert_lerobot_action_to_leisaac(
                bridge.applied_motor[None, :]
            )
            action = torch.from_numpy(sim_action).float().to(env.device)

            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, task_type)

            obs_dict, _, terminated, _, _ = env.step(action)

            # Success resets the episode inside step(). Report it, and clear
            # the commands so the reset arm holds its new pose
            if bool(terminated[0]):
                bridge.get_logger().info(
                    "TASK SUCCESS: success termination fired; env auto-reset"
                )
                bridge.success_pub.publish(String(data="YES"))
                bridge.pending_cmd = None
                bridge.applied_motor = None
                bridge.target_queue.clear()

            stamp = bridge.get_clock().now().to_msg()
            bridge.publish_joint_state(joint_pos, stamp)
            if step_count % camera_every == 0:
                for key in camera_keys:
                    image = policy_obs[key].cpu().numpy().astype(np.uint8)[0]
                    bridge.publish_image(key, image, stamp)

        rclpy.spin_once(bridge, timeout_sec=0.0)
        step_count += 1

        elapsed = time.perf_counter() - loop_start
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

    bridge.destroy_node()
    rclpy.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
