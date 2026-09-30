"""VLA component driving the LeIsaac SO101 pick-orange scene.

Reads the topics published by scripts/leisaac_ros2_bridge.py (joint values in
SO101 motor units) and sends observations to the LeRobot policy server, which
serves a GR00T N1.7 fine-tune of the LightwheelAI/leisaac-pick-orange dataset.
The goal ends after a fixed timestep budget.

Run it in your ROS 2 environment, not the Isaac venv:

    python3 recipes/vla_sim_recipe.py

Then open https://localhost:5001 (accept the self-signed certificate once) and
start the task with the dataset's exact wording:

    Grab orange and place into plate

Or from the command line:

    ros2 action send_goal /vla_sim/manipulate_with_vla \\
        automatika_embodied_agents/action/VisionLanguageAction \\
        "{task: 'Grab orange and place into plate'}"

Variants: vla_sim_recipe_sim_success.py ends the goal on the sim's success
marker, vla_sim_recipe_vlm_success.py on a local VLM judge's verdict.
"""

import os

from agents.clients import LeRobotClient
from agents.components import VLA
from agents.config import VLAConfig
from agents.models import LeRobotPolicy
from agents.ros import Launcher, Topic

# Set these to local copies to avoid downloading from the HuggingFace hub
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

# SO101 motor-unit ranges, the space every bridge topic uses
SO101_MOTOR_LIMITS = {
    joint: {"lower": -100.0, "upper": 100.0} for joint in SO101_JOINTS[:-1]
}
SO101_MOTOR_LIMITS["gripper"] = {"lower": 0.0, "upper": 100.0}

joint_states = Topic(name="/so101/joint_states", msg_type="JointState")
front_camera = Topic(name="/so101/front/image_raw", msg_type="Image")
wrist_camera = Topic(name="/so101/wrist/image_raw", msg_type="Image")
joint_cmd = Topic(name="/so101/joint_cmd", msg_type="JointState")

# Trained with chunk_size=16, so actions_per_chunk must be at most 16
policy = LeRobotPolicy(
    name="pick_orange_gr00t",
    policy_type="groot",
    checkpoint=CHECKPOINT,
    dataset_info_file=DATASET_INFO,
    actions_per_chunk=16,
)

# Alternative: a SmolVLA fine-tune, less accurate but needs no server patch
# policy = LeRobotPolicy(
#     name="pick_orange_smolvla",
#     policy_type="smolvla",
#     checkpoint="aleph-ra/smolvla_finetune_pick_orange_20000",
#     dataset_info_file=DATASET_INFO,
# )

model_client = LeRobotClient(model=policy, host="127.0.0.1", port=8080)

config = VLAConfig(
    joint_names_map={f"{joint}.pos": joint for joint in SO101_JOINTS},
    camera_inputs_map={"front": front_camera, "wrist": wrist_camera},
    # Caps outgoing commands. The URDF gives the same limits with
    # robot_urdf_file="assets/so101_new_calib.urdf" and
    # policy_action_units="normalized"
    joint_limits=SO101_MOTOR_LIMITS,
    # Play actions at the dataset's cadence, see "Timing configuration" in the
    # README
    observation_sending_rate=0.55,
    action_sending_rate=10.0,
    # Run one chunk at a time: this checkpoint predicts relative actions, so
    # chunks from different inference calls must not be blended
    aggregate_fn_name="latest_only",
)

vla = VLA(
    inputs=[joint_states, front_camera, wrist_camera],
    outputs=[joint_cmd],
    model_client=model_client,
    config=config,
    component_name="vla_sim",
)
# 1800 steps at 10 Hz is about 60 s of sim time, three times an average
# demonstration
vla.set_termination_trigger(mode="timesteps", max_timesteps=1800)

launcher = Launcher()
launcher.enable_ui(
    inputs=[vla.ui_main_action_input],
    outputs=[front_camera],
)
launcher.add_pkg(components=[vla])
launcher.bringup()
