#!/usr/bin/env python3
"""Make leisaac's own LeRobot policy client work with lerobot >= 0.6 servers.

Only needed for leisaac's scripts/evaluation/policy_inference.py; the
embodied-agents pipeline does not use this client. Safe to run repeatedly.

    python patches/apply_leisaac_patches.py --leisaac <path-to-leisaac-checkout>

A. lerobot 0.6 moved the async-inference helpers, so the server could not
   unpickle what the client sends.
B. Adds RemotePolicyConfig.rename_map, which the 0.6 server reads.
C. Adds gRPC deadlines, and stops the client from giving up on observations
   after one empty GetActions reply, which froze the arm.
"""

import argparse
import sys
from pathlib import Path

EDITS = {
    "source/leisaac/leisaac/policy/lerobot/__init__.py": [
        (
            'helpers_path = "lerobot.scripts.server.helpers"',
            '# PATCH(embodied-agents-sim-setup): lerobot >= 0.6 moved these helpers.\n'
            'helpers_path = "lerobot.async_inference.helpers"',
            'lerobot.async_inference.helpers',
        ),
    ],
    "source/leisaac/leisaac/policy/lerobot/helpers.py": [
        (
            "from dataclasses import dataclass\nfrom enum import Enum",
            "from dataclasses import dataclass, field\nfrom enum import Enum",
            "from dataclasses import dataclass, field",
        ),
        (
            """@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu\"""",
            """@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    # PATCH(embodied-agents-sim-setup): lerobot >= 0.6 added rename_map and
    # the server reads it unconditionally.
    rename_map: dict[str, str] = field(default_factory=dict)""",
            "rename_map: dict[str, str]",
        ),
    ],
    "source/leisaac/leisaac/policy/service_policy_clients.py": [
        (
            "        _ = self.stub.SendObservations(observation_iterator)",
            "        # PATCH(embodied-agents-sim-setup): a deadline, so a stalled\n"
            "        # stream raises instead of hanging the evaluation\n"
            "        _ = self.stub.SendObservations(observation_iterator, timeout=10.0)",
            "SendObservations(observation_iterator, timeout=10.0)",
        ),
        (
            "        actions_chunk = self.stub.GetActions(services_pb2.Empty())",
            "        actions_chunk = self.stub.GetActions(services_pb2.Empty(), timeout=10.0)",
            "GetActions(services_pb2.Empty(), timeout=10.0)",
        ),
        (
            """    def get_action(self, observation_dict: dict) -> torch.Tensor:
        if not self.skip_send_observation:
            self._send_observation(observation_dict)
        action_chunk = self._receive_action()""",
            """    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # PATCH(embodied-agents-sim-setup): always send the observation and
        # poll for the chunk with a bounded retry; hold the last action only
        # if the call fails. One empty reply used to stop observations for good
        try:
            self._send_observation(observation_dict)
            action_chunk = None
            for _ in range(50):  # up to ~10 s for inference to complete
                action_chunk = self._receive_action()
                if action_chunk is not None:
                    break
                time.sleep(0.2)
        except grpc.RpcError as e:
            print(f"[CLIENT] gRPC call failed ({e.code()}), holding last action and resyncing")
            if self.last_action is not None:
                return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]
            raise""",
            "bounded retry; hold the last action",
        ),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leisaac", required=True, help="path to the leisaac checkout")
    parser.add_argument("--check", action="store_true", help="report state, change nothing")
    args = parser.parse_args()
    root = Path(args.leisaac)

    ok = True
    for rel, edits in EDITS.items():
        path = root / rel
        if not path.exists():
            print(f"[skip] {rel}: not found (leisaac layout drift?)")
            ok = False
            continue
        src = path.read_text()
        changed = False
        for old, new, marker in edits:
            if marker in src:
                print(f"[ok]   {rel}: '{marker[:48]}' already applied")
            elif old in src:
                if not args.check:
                    src = src.replace(old, new, 1)
                    changed = True
                print(f"[{'need' if args.check else 'done'}] {rel}: '{marker[:48]}'")
            else:
                print(f"[??]   {rel}: anchor for '{marker[:48]}' not found, inspect manually")
                ok = False
        if changed:
            path.write_text(src)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
