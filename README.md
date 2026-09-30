# Embodied Agents Simulation Setup (Isaac Sim + LeIsaac)

A ready-to-run simulation for testing the **VLA component** of [embodied-agents](https://github.com/automatika-robotics/embodied-agents): **NVIDIA Isaac Sim / Isaac Lab**, the [LeIsaac](https://github.com/LightwheelAI/leisaac) SO101 environments, and a GR00T N1.7 policy served by the LeRobot async policy server.

```
┌────────────────────────────┐   ROS2 topics    ┌─────────────────────┐   gRPC    ┌──────────────────────┐
│ Isaac Sim (LeIsaac scene)  │ ───────────────► │ embodied-agents     │ ────────► │ LeRobot Async Policy │
│ scripts/                   │  /so101/joint_states  VLA component    │  obs      │ Server (groot)       │
│   leisaac_ros2_bridge.py   │  /so101/front/image_raw               │           │                      │
│                            │  /so101/wrist/image_raw               │ ◄──────── │                      │
│                            │ ◄─────────────── │                     │  actions  └──────────────────────┘
│                            │  /so101/joint_cmd│                     │
└────────────────────────────┘                  └─────────────────────┘
```

An SO101 arm moves oranges from a kitchen counter to a plate. Scene, data and policy match: the `LeIsaac-SO101-PickOrange-v0` scene, the `LightwheelAI/leisaac-pick-orange` dataset recorded in it, and [`aleph-ra/gr00t17_pick_orange_lora`](https://huggingface.co/aleph-ra/gr00t17_pick_orange_lora), a GR00T N1.7 fine-tune on that dataset.

## Prerequisites

- Ubuntu 22.04 (ROS 2 Humble) or later
- An NVIDIA RTX-class GPU, driver 535 or later, ideally 24 GB of VRAM (see [GPU budget](#gpu-budget))
- About 80 GB of free disk for a fresh install
- Python 3.11 and 3.12, or [`uv`](https://docs.astral.sh/uv/), which the installer uses to fetch a missing interpreter
- HuggingFace access to the gated [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) model, which every GR00T checkpoint loads (request it, then `hf auth login`)
- EMOS, or just an embodied-agents ROS 2 workspace, to run the recipes

## Quickstart

```bash
./doctor.sh        # 1. preflight checks, with a fix for each problem
./setup.sh         # 2. install everything (safe to re-run after a failure)
make server        # 3. terminal A: LeRobot policy server
make bridge        # 4. terminal B: Isaac Sim + ROS 2 bridge (GUI)
make status        # 5. check that topics are flowing
```

Then run a recipe, see [Running an evaluation](#running-an-evaluation).

Everything installs under `SIM_ROOT`, by default this repo's parent directory. Versions are pinned in [`versions.env`](versions.env); read [Version pins and patches](#version-pins-and-patches) before changing one.

> **Note:** the first Isaac Sim launch compiles shaders and can take 10–20 minutes with an unresponsive window. Later launches take 1–2 minutes.

`make help` lists the other targets, such as `make reset`, `make smoke` and `make clean-procs`.

## Running an evaluation

With `make server` and `make bridge` running, start a recipe from your ROS 2 environment, **not** the Isaac venv:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source ~/ros_ws/install/setup.bash
python3 recipes/vla_sim_recipe.py
```

Once its components have started, open the web UI at **`https://localhost:5001`** (accept its self-signed certificate once) and start the task:

```
Grab orange and place into plate
```

Or send the same goal from the command line:

```bash
ros2 action send_goal /vla_sim/manipulate_with_vla \
    automatika_embodied_agents/action/VisionLanguageAction \
    "{task: 'Grab orange and place into plate'}"
```

Reset the scene between episodes with `make reset`, or `R` in the Isaac window.

> **Use the task string exactly as written.** A reworded instruction degrades the policy.

### Recipe variants

| Recipe                          | Goal ends on                                                 | UI panels                      |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------ |
| `vla_sim_recipe.py`             | a fixed timestep budget                                      | goal control, front camera     |
| `vla_sim_recipe_sim_success.py` | an event on the simulator's success marker                   | goal control, camera, marker   |
| `vla_sim_recipe_vlm_success.py` | an event on a local VLM judge watching the camera, no oracle | goal control, camera, verdicts |

The event recipes end the goal with `success: true` when the task is judged complete, and otherwise at the timestep budget. In your own success events, use `on_change=True` only on topics that publish a stream of verdicts, like the VLM judge. Leave it out for one-shot markers like `/so101/task_success`: an edge-triggered event needs a negative reading before it can fire.

### The VLM success judge

The VLM recipe shows the front camera to a local model served by [Ollama](https://ollama.com) and asks whether every orange is in the plate.

- **VRAM.** Ollama loads a model on the GPU only if it fits in the _remaining_ VRAM. Next to Isaac and the policy server on a 24 GB card, only small models such as `qwen2.5vl:3b` fit, and they may still land on the CPU.
- **CPU load.** A judge on the CPU competes with the sim loop and can upset the policy's timing, so the recipe limits it to 2 threads and one verdict every 120 s. With GPU room, a few seconds is fine.

## Timing configuration

Actions must play back at the dataset's recording rate **in simulation time**: 30 fps. The bridge advances 1/60 s of sim time per step at about 20 steps per second of wall time, so the sim runs at a third of real time. Hence:

- `action_sending_rate=10.0`: each action is held for about 2 sim steps, which gives 30 fps in sim time
- `observation_sending_rate=0.55`: a full 16-action chunk, plus inference time, runs before the next chunk replaces it
- `aggregate_fn_name="latest_only"`: one chunk at a time, never blended, since this checkpoint predicts actions relative to the state of their own chunk

If your bridge steps at another rate (`make status` shows it), scale the rates: `action_sending_rate ≈ dataset_fps × step_rate / 60` and `observation_sending_rate ≈ 1 / (chunk_size / action_rate + inference_time)`. Playing actions too fast, or replacing chunks before they finish, makes motion fast and erratic.

## How the bridge works

`make bridge` launches Isaac Sim with the kitchen scene and exposes it over ROS 2, using Isaac Sim's bundled ROS 2 libraries if the system `rclpy` targets another Python version.

| Topic / service          | Type                     | Direction | Notes                                                   |
| ------------------------ | ------------------------ | --------- | ------------------------------------------------------- |
| `/so101/joint_states`    | `sensor_msgs/JointState` | sim → out | positions in motor units                                |
| `/so101/front/image_raw` | `sensor_msgs/Image`      | sim → out | 480×640 rgb8                                            |
| `/so101/wrist/image_raw` | `sensor_msgs/Image`      | sim → out | 480×640 rgb8                                            |
| `/so101/joint_cmd`       | `sensor_msgs/JointState` | in → sim  | absolute positions, motor units                         |
| `/so101/reset`           | `std_srvs/Trigger`       | service   | resets the episode                                      |
| `/so101/task_success`    | `std_msgs/String`        | sim → out | `"YES"`, published once when the scene's success is met |

Commands are applied one per simulation step in the order they arrive, as in LeIsaac's own evaluation client.

### Units: motor space, not radians

The policy does **not** work in radians. LeIsaac datasets use the SO101 _motor_ space: each joint's range mapped linearly to [-100, 100], the gripper's to [0, 100]. The bridge converts, so **every ROS topic here carries motor units**.

| Joint         | Joint limits (deg) | Motor units |
| ------------- | ------------------ | ----------- |
| shoulder_pan  | (-110, 110)        | (-100, 100) |
| shoulder_lift | (-100, 100)        | (-100, 100) |
| elbow_flex    | (-100, 90)         | (-100, 100) |
| wrist_flex    | (-95, 95)          | (-100, 100) |
| wrist_roll    | (-160, 160)        | (-100, 100) |
| gripper       | (-10, 100)         | (0, 100)    |

On `/so101/joint_states`, values like `65.8` are motor units; values within ±1.5 are radians, which the policy was never trained on. It expects cameras `front` and `wrist` (480×640×3 RGB), state keys `shoulder_pan.pos` to `gripper.pos`, and the task string.

## Version pins and patches

[`versions.env`](versions.env) pins every component, and each pin matters:

| Pin                 | Why                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------- |
| IsaacLab `v2.3.0`   | `main` (6.x) needs Isaac Sim 6 and does not install against 5.1.                                                      |
| isaacsim `5.1.0.0`  | The release LeIsaac targets (cp311 wheels).                                                                           |
| torch `2.7.0+cu128` | Required by isaacsim 5.1 and IsaacLab 2.3.                                                                            |
| lerobot `0.6.1`     | GR00T needs 0.6.0 or later. Before upgrading, check for [PR #3981](https://github.com/huggingface/lerobot/pull/3981). |
| leisaac commit      | The commit the bridge was validated against.                                                                          |

`setup.sh` applies two sets of patches, both safe to re-run; re-run them after reinstalling lerobot:

- **lerobot policy server** (`patches/apply_lerobot_patches.py`): postprocesses whole action chunks, as GR00T needs (PR #3981), and frees the loaded policy before loading a new one, so recipe restarts don't exhaust GPU memory.
- **leisaac client** (`patches/apply_leisaac_patches.py`): makes LeIsaac's own evaluation client work with lerobot 0.6. The embodied-agents pipeline does not use it.

## GPU budget

On a single 24 GB card:

| Process                               | VRAM    |
| ------------------------------------- | ------- |
| LeRobot server with GR00T N1.7 loaded | ~9.5 GB |
| Isaac Sim + kitchen scene + cameras   | ~6 GB   |
| Headroom for inference                | rest    |

Start the policy server before Isaac. On a smaller card, serve the policy from another machine, or use the less accurate SmolVLA fine-tune `aleph-ra/smolvla_finetune_pick_orange_20000`, which needs no server patch.

## Troubleshooting

**The arm never moves and the server returns no actions.**
The lerobot patch is missing, for instance after a lerobot reinstall. `./setup.sh --verify` reports it, and `$SIM_ROOT/lerobot-venv/bin/python patches/apply_lerobot_patches.py` re-applies it.

**Actions look random and the arm flails.**
Check that `/so101/joint_states` shows motor units, not radians, that the task string is exact, and that both cameras arrive.

**Motion is fast and jerky.**
The cadence is off, see [Timing configuration](#timing-configuration). Usually the CPU is in its power-saving governor, which `./doctor.sh` detects and `sudo cpupower frequency-set -g performance` fixes. A VLM judge on the CPU can also slow the sim; throttle it.

**A goal is rejected with "action client not ready".**
Wait until the logs report that every component has started.

**Odd behaviour after restarting recipes.**
Leftover processes hold ports and node names. Run `make clean-procs`, then `./doctor.sh`.

**Downloads keep failing on a flaky network.**
Run `./setup.sh` again; it resumes where it stopped.

**The gripper never closes.**
The bridge calls `dynamic_reset_gripper_effort_limit_sim` every step, as LeIsaac's evaluation loop does. Keep that call if you change the loop.

## License

This repository is released under the [MIT License](LICENSE), copyright Automatika Robotics.

`assets/so101_new_calib.urdf` is the SO101 arm model from The Robot Studio's [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) repository (`Simulation/SO101/so101_new_calib.urdf`, commit `385e8d7`), included unmodified under the Apache License 2.0; see [`assets/LICENSE.so101_urdf`](assets/LICENSE.so101_urdf).

The scripts in `patches/` modify installed copies of [lerobot](https://github.com/huggingface/lerobot) and [LeIsaac](https://github.com/LightwheelAI/leisaac), both Apache License 2.0, and quote short passages of their source to locate the changes. No other code from either project is included here.
