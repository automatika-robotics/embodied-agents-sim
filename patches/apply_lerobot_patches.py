#!/usr/bin/env python3
"""Apply two local patches to an installed lerobot 0.6.x policy server.

Safe to run repeatedly and after any lerobot reinstall. Use the lerobot venv:

    <lerobot-venv>/bin/python patches/apply_lerobot_patches.py [--check]

1. Full-chunk postprocessing (lerobot PR #3981). The server postprocesses a
   chunk one action at a time, which GR00T N1.7 checkpoints with relative
   actions reject, so it returns no actions at all. Skipped once lerobot
   ships the fix.
2. Release the loaded policy before loading a new one. Otherwise every
   recipe restart leaves a copy on the GPU, about 9 GiB for GR00T.
"""

import argparse
import sys
from pathlib import Path

try:
    import lerobot.async_inference.policy_server as ps_mod
except ImportError:
    sys.exit("ERROR: run this with the lerobot venv's python (lerobot not importable)")

TARGET = Path(ps_mod.__file__)

PER_STEP_LOOP = """\
        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)"""

FULL_CHUNK_FIRST = """\
        try:
            # Postprocess the full (B, chunk_size, action_dim) chunk in one pass,
            # as GR00T N1.7 with relative actions requires
            action_tensor = self.postprocessor(action_tensor).squeeze(0)
        except Exception:
            # Per-step postprocessing for pipelines that need it
            processed_actions = []
            for i in range(chunk_size):
                # Extract action at timestep i: (B, action_dim)
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)

            # Stack back to (B, chunk_size, action_dim), then remove batch dim
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)"""

LOAD_ANCHOR = """\
        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()"""

RELEASE_BLOCK = """\
        policy_class = get_policy_class(self.policy_type)

        # PATCH(embodied-agents-sim-setup): free the loaded policy first, or
        # every client re-init keeps another copy on the GPU
        if getattr(self, "policy", None) is not None:
            del self.policy
            self.preprocessor = None
            self.postprocessor = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Released previously loaded policy before re-init")

        start = time.perf_counter()"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="report state, change nothing")
    args = parser.parse_args()

    src = TARGET.read_text()
    changed = False

    # --- Patch 1 -----------------------------------------------------------
    if "Postprocess the full" in src:
        print(f"[1] full-chunk postprocessing: already applied (or upstream fix present)")
    elif PER_STEP_LOOP in src:
        if not args.check:
            src = src.replace(PER_STEP_LOOP, FULL_CHUNK_FIRST, 1)
            changed = True
        print(f"[1] full-chunk postprocessing: {'NEEDED' if args.check else 'applied'}")
    else:
        print("[1] WARNING: neither the per-step loop nor the fix found (lerobot "
              "version drift?); inspect _predict_action_chunk manually")

    # --- Patch 2 -----------------------------------------------------------
    if "Released previously loaded policy" in src:
        print(f"[2] release-policy-on-reinit: already applied")
    elif LOAD_ANCHOR in src:
        if not args.check:
            src = src.replace(LOAD_ANCHOR, RELEASE_BLOCK, 1)
            changed = True
        print(f"[2] release-policy-on-reinit: {'NEEDED' if args.check else 'applied'}")
    else:
        print("[2] WARNING: SendPolicyInstructions anchor not found (lerobot "
              "version drift?); inspect the policy-load path manually")

    if changed:
        TARGET.write_text(src)
        print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
