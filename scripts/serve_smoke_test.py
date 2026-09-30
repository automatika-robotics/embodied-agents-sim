"""Smoke test a checkpoint through the LeRobot async policy server.

Uses the same gRPC protocol as the embodied-agents LeRobotClient, sends one
frame from the training dataset and checks the returned action chunk. Needs
only the lerobot venv and a running server (`make server`).
"""

import argparse
import pickle
import time

import grpc
import numpy as np

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks

parser = argparse.ArgumentParser()
parser.add_argument("--policy-type", default="groot")
parser.add_argument("--checkpoint", required=True, help="HF repo id or local path")
parser.add_argument("--dataset-repo", default="LightwheelAI/leisaac-pick-orange")
parser.add_argument("--dataset-root", default=None, help="Local dataset root override")
parser.add_argument("--frame", type=int, default=100, help="Dataset frame index to send")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--actions-per-chunk", type=int, default=16)
parser.add_argument("--device", default="cuda")
parser.add_argument("--task", default=None, help="Override task instruction")
args = parser.parse_args()

# Motor-unit sanity ranges for the SO101 (gripper is 0..100)
MOTOR_RANGE = (-120.0, 120.0)

print(f"Loading dataset {args.dataset_repo} ...")
dataset = LeRobotDataset(args.dataset_repo, root=args.dataset_root, episodes=[0])
features = {
    key: dict(feature)
    for key, feature in dataset.meta.features.items()
    if key == "observation.state" or key.startswith("observation.images.")
}
state_names = features["observation.state"]["names"]
camera_keys = [
    key.removeprefix("observation.images.")
    for key in features
    if key.startswith("observation.images.")
]
print(f"state names: {state_names}")
print(f"cameras: {camera_keys}")

item = dataset[args.frame]
task = args.task or item.get("task") or "Grab orange and place into plate"
print(f"task: {task!r}")

# Built the way the embodied-agents VLA component builds it
raw_observation = {"task": task}
for cam in camera_keys:
    frame = item[f"observation.images.{cam}"]  # CHW float32 in [0, 1]
    raw_observation[cam] = (
        (frame * 255).byte().permute(1, 2, 0).contiguous().numpy()
    )
state = item["observation.state"].numpy()
for i, name in enumerate(state_names):
    raw_observation[name] = float(state[i])

ground_truth = item["action"].numpy()

channel = grpc.insecure_channel(f"{args.host}:{args.port}")
stub = services_pb2_grpc.AsyncInferenceStub(channel)

print("Pinging server ...")
stub.Ready(services_pb2.Empty())

print(f"Initializing policy '{args.policy_type}' from {args.checkpoint} ...")
config = RemotePolicyConfig(
    policy_type=args.policy_type,
    pretrained_name_or_path=args.checkpoint,
    lerobot_features=features,
    actions_per_chunk=args.actions_per_chunk,
    device=args.device,
)
stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(config)))
print("Policy initialized on server.")

observation = TimedObservation(
    timestamp=time.time(), timestep=0, observation=raw_observation, must_go=True
)
obs_bytes = pickle.dumps(observation)
stub.SendObservations(
    send_bytes_in_chunks(obs_bytes, services_pb2.Observation, silent=True)
)
print("Observation sent, polling for actions ...")

actions = None
for attempt in range(30):
    response = stub.GetActions(services_pb2.Empty())
    if response.data:
        actions = pickle.loads(response.data)
        break
    time.sleep(1.0)

if not actions:
    raise SystemExit("FAIL: no actions received from the server within 30s")

chunk = np.stack([timed.action.numpy().squeeze() for timed in actions])
print(f"\nReceived {chunk.shape[0]} actions of dim {chunk.shape[-1]}")
print(f"per-dim min: {np.round(chunk.min(axis=0), 2)}")
print(f"per-dim max: {np.round(chunk.max(axis=0), 2)}")
print(f"ground-truth action at frame {args.frame}: {np.round(ground_truth, 2)}")
print(f"first predicted action:                {np.round(chunk[0], 2)}")
print(f"|first - ground truth|:                {np.round(np.abs(chunk[0] - ground_truth), 2)}")

ok = True
if not np.all(np.isfinite(chunk)):
    print("FAIL: non-finite values in action chunk")
    ok = False
if chunk.min() < MOTOR_RANGE[0] or chunk.max() > MOTOR_RANGE[1]:
    print("FAIL: actions outside motor-unit ranges — check normalization")
    ok = False
if np.abs(chunk).max() < 5.0:
    print(
        "WARN: all actions are tiny (|a| < 5). If the policy was trained with "
        "use_relative_actions=true, the server-side postprocessor may not be "
        "resolving relative actions to absolute targets — report this."
    )

print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
