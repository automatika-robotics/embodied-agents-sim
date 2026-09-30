"""VLA recipe whose goal ends when a local VLM judges the task done.

A small VLM (qwen2.5vl:3b through Ollama) looks at the front camera on a
timer and answers whether every orange is in the plate. An event on its
answer ends the goal, with the timestep budget as the limit for failed
episodes. No simulator oracle is involved, so the same pattern works on a
real robot.

    front camera --> VLM judge (timed) --> /vla_sim/success_check
                                                | event: contains "YES"
    VLA goal <----------- ends ----------------+

    python3 recipes/vla_sim_recipe_vlm_success.py

Start the task from https://localhost:5001 with the wording
"Grab orange and place into plate", or send the goal as in vla_sim_recipe.py.
The UI also shows the judge's verdicts.
"""

import os

from agents.clients import LeRobotClient, OllamaClient
from agents.components import VLA, VLM
from agents.config import VLAConfig
from agents.models import LeRobotPolicy, OllamaModel
from agents.ros import Event, FixedInput, Launcher, Topic

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

# The judge. On a 24 GB card shared with Isaac and the policy server it may
# run on the CPU, where it competes with the sim loop: num_thread and the long
# trigger period keep that load small. See "The VLM success judge" in the
# README
judge_model = OllamaModel(
    name="success_judge_vlm",
    checkpoint="qwen2.5vl:3b",
    options={"num_ctx": 4096, "num_predict": 5, "num_thread": 2},
)
judge_client = OllamaClient(model=judge_model, inference_timeout=240)

judge_prompt = FixedInput(
    name="judge_prompt",
    msg_type="String",
    fixed=(
        "Look at the white and blue bowl on the kitchen counter. Are ALL three of the "
        "oranges inside that bowl, with none left on the counter? "
        "Answer with exactly one word: YES or NO."
    ),
)

success_check = Topic(name="/vla_sim/success_check", msg_type="String")

judge = VLM(
    inputs=[judge_prompt, front_camera],
    outputs=[success_check],
    model_client=judge_client,
    # Seconds between verdicts. A few seconds is fine on a GPU with room
    trigger=120.0,
    component_name="success_judge",
)

# The judge answers on every tick, so the event fires on the change to YES
# rather than on every YES
success_event = Event(
    success_check.msg.data.contains("YES"),
    on_change=True,
)
vla.set_termination_trigger(
    mode="event", stop_event=success_event, max_timesteps=1800
)

launcher = Launcher()
launcher.enable_ui(
    inputs=[vla.ui_main_action_input],
    outputs=[success_check, front_camera],
)
launcher.add_pkg(components=[vla, judge])
launcher.bringup()
