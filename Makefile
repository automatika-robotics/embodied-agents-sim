# Task runner for the embodied-agents sim stack. See the README for the workflow.
SHELL := /bin/bash
HERE  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SIM_ROOT ?= $(abspath $(HERE)/..)
ROS_SETUP := /opt/ros/$$(ls /opt/ros 2>/dev/null | head -1)/setup.bash

.PHONY: help doctor setup verify server bridge bridge-headless smoke status reset clean-procs

help:
	@echo "Targets:"
	@echo "  doctor          preflight environment checks"
	@echo "  setup           install everything (idempotent, resumable)"
	@echo "  verify          re-run setup verification only"
	@echo "  server          run the LeRobot policy server (terminal A)"
	@echo "  bridge          run Isaac Sim + ROS2 bridge with GUI (terminal B)"
	@echo "  bridge-headless same, without GUI"
	@echo "  smoke           policy-server smoke test (server must be running)"
	@echo "  status          topics, rates, port and VRAM overview"
	@echo "  reset           reset the sim episode (via /so101/reset service)"
	@echo "  clean-procs     kill stale servers/bridges/zombie recipe Launchers"

doctor:
	@bash $(HERE)doctor.sh

setup:
	@bash $(HERE)setup.sh

verify:
	@bash $(HERE)setup.sh --verify

server:
	@bash $(HERE)scripts/run_server.sh

bridge:
	@bash $(HERE)scripts/run_bridge.sh

bridge-headless:
	@HEADLESS=1 bash $(HERE)scripts/run_bridge.sh

# CHECKPOINT can be an HF repo id or a local pretrained_model directory.
CHECKPOINT ?= aleph-ra/gr00t17_pick_orange_lora
smoke:
	@source $(HERE)versions.env && \
	HF_HUB_DOWNLOAD_TIMEOUT=60 $(SIM_ROOT)/lerobot-venv/bin/python \
	    $(HERE)scripts/serve_smoke_test.py --policy-type=groot \
	    --checkpoint="$(CHECKPOINT)" --dataset-repo="$$DATASET_REPO"

status:
	@echo "== port 8080 ==" && (ss -tlnp 2>/dev/null | grep 8080 || echo "  no listener")
	@echo "== GPU ==" && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/  /' || true
	@echo "== ROS graph ==" && source $(ROS_SETUP) && timeout 10 ros2 node list 2>/dev/null | sed 's/^/  /' || true
	@echo "== topic rates (5s each) ==" && source $(ROS_SETUP) && \
	  for t in /so101/joint_states /so101/front/image_raw; do \
	    echo -n "  $$t: "; timeout 6 ros2 topic hz $$t 2>/dev/null | head -1 || echo "no data"; \
	  done

reset:
	@source $(ROS_SETUP) && ros2 service call /so101/reset std_srvs/srv/Trigger "{}"

clean-procs:
	@echo "killing stale policy servers, bridges and zombie recipe Launchers..."
	-@pkill -9 -f lerobot.async_inference.policy_server 2>/dev/null || true
	-@pkill -9 -f leisaac_ros2_bridge 2>/dev/null || true
	-@ps -eo pid,comm | awk '$$2=="Launcher"{print $$1}' | xargs -r kill -9 2>/dev/null || true
	-@pkill -9 -f ui_node 2>/dev/null || true
	@sleep 2
	@source $(ROS_SETUP) && ros2 daemon stop >/dev/null 2>&1 || true
	@echo "done; port 8080 listeners now: $$(ss -tln 2>/dev/null | grep -c 8080 || echo 0)"
