"""VLA recipe whose goal ends on the sim's own success marker.

The bridge publishes "YES" on /so101/task_success when the scene's success
condition fires: every orange in the plate and the arm back at rest. An event
on that topic ends the goal there instead of at the timestep budget, which
stays as the limit for failed episodes. Otherwise the recipe matches
vla_sim_recipe.py.

    python3 recipes/vla_sim_recipe_sim_success.py

Start the task from https://localhost:5001 with the wording
"Grab orange and place into plate", or send the goal as in vla_sim_recipe.py.
"""

import os

from agents.clients import LeRobotClient
from agents.components import VLA
from agents.config import VLAConfig
from agents.models import LeRobotPolicy
from agents.ros import Event, Launcher, Topic

CHECKPOINT = os.environ.get(
    "PICK_ORANGE_CHECKPOINT", "aleph-ra/gr00t17_pick_orange_lora"
)
DATASET_INFO = os.environ.get(
    "PICK_ORANGE_DATASET_INFO",
    "https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange/resolve/main/meta/info.json",
)

SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

SO101_MOTOR_LIMITS = {
    joint: {"lower": -100.0, "upper": 100.0} for joint in SO101_JOINTS[:-1]
}
SO101_MOTOR_LIMITS["gripper"] = {"lower": 0.0, "upper": 100.0}

joint_states = Topic(name="/so101/joint_states", msg_type="JointState")
front_camera = Topic(name="/so101/front/image_raw", msg_type="Image")
wrist_camera = Topic(name="/so101/wrist/image_raw", msg_type="Image")
joint_cmd = Topic(name="/so101/joint_cmd", msg_type="JointState")
task_success = Topic(name="/so101/task_success", msg_type="String")

policy = LeRobotPolicy(
    name="pick_orange_gr00t",
    policy_type="groot",
    checkpoint=CHECKPOINT,
    dataset_info_file=DATASET_INFO,
    actions_per_chunk=16,
)
model_client = LeRobotClient(model=policy, host="127.0.0.1", port=8080)

config = VLAConfig(
    joint_names_map={f"{joint}.pos": joint for joint in SO101_JOINTS},
    camera_inputs_map={"front": front_camera, "wrist": wrist_camera},
    joint_limits=SO101_MOTOR_LIMITS,
    observation_sending_rate=0.55,
    action_sending_rate=10.0,
    aggregate_fn_name="latest_only",
)

vla = VLA(
    inputs=[joint_states, front_camera, wrist_camera],
    outputs=[joint_cmd],
    model_client=model_client,
    config=config,
    component_name="vla_sim",
)

# The marker is published once per success, so the event has no on_change:
# an edge-triggered event would need a "NO" before it could fire
success_event = Event(task_success.msg.data.contains("YES"))
vla.set_termination_trigger(
    mode="event", stop_event=success_event, max_timesteps=1800
)

launcher = Launcher()
launcher.enable_ui(
    inputs=[vla.ui_main_action_input],
    outputs=[task_success, front_camera],
)
launcher.add_pkg(components=[vla])
launcher.bringup()
