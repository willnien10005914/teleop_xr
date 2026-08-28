"""Heart gesture IK targets for bimanual robots."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from teleop_xr.ik.control_mode import ControlMode

if TYPE_CHECKING:
    from teleop_xr import Teleop
    from teleop_xr.ik.controller import IKController
    from teleop_xr.ik.robot import BaseRobot


# Bimanual targets in the robot base frame (X forward, Y left, Z up).
_HEART_LEFT = {
    "position": {"x": 0.28, "y": 0.06, "z": 0.58},
    "orientation": {"w": 0.85, "x": 0.35, "y": -0.35, "z": 0.15},
}
_HEART_RIGHT = {
    "position": {"x": 0.28, "y": -0.06, "z": 0.58},
    "orientation": {"w": 0.85, "x": -0.35, "y": -0.35, "z": -0.15},
}


def get_heart_pose_targets() -> dict[str, dict[str, dict[str, float]]]:
    """Return absolute EE targets that form a heart pose with both arms."""

    return {"left": dict(_HEART_LEFT), "right": dict(_HEART_RIGHT)}


def run_heart_gesture(
    controller: "IKController",
    robot: "BaseRobot",
    teleop: "Teleop",
    q_current: np.ndarray,
    teleop_loop: Any | None,
    logger: Any,
    *,
    steps: int = 30,
    hold_s: float = 1.5,
    step_sleep_s: float = 0.04,
) -> np.ndarray:
    """Animate both end effectors into a heart pose via IK, then hold briefly."""

    original_mode = controller.get_mode()
    original_mode_value = getattr(original_mode, "value", original_mode)
    q_next = np.array(q_current)

    try:
        controller.set_mode(ControlMode.EE_ABSOLUTE)
        logger.info("Heart gesture started")

        heart_targets = get_heart_pose_targets()
        for step_idx in range(steps):
            alpha = (step_idx + 1) / steps
            blended: dict[str, dict[str, dict[str, float]]] = {}
            current_fk = robot.forward_kinematics(jnp.asarray(q_next))
            for frame, target in heart_targets.items():
                if frame not in current_fk:
                    continue
                start_pos = np.asarray(current_fk[frame].translation(), dtype=float)
                start_wxyz = np.asarray(current_fk[frame].rotation().wxyz, dtype=float)
                goal_pos = np.array(
                    [target["position"]["x"], target["position"]["y"], target["position"]["z"]],
                    dtype=float,
                )
                goal_wxyz = np.array(
                    [
                        target["orientation"]["w"],
                        target["orientation"]["x"],
                        target["orientation"]["y"],
                        target["orientation"]["z"],
                    ],
                    dtype=float,
                )
                pos = start_pos + alpha * (goal_pos - start_pos)
                wxyz = start_wxyz + alpha * (goal_wxyz - start_wxyz)
                wxyz = wxyz / np.linalg.norm(wxyz)
                blended[frame] = {
                    "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                    "orientation": {
                        "w": float(wxyz[0]),
                        "x": float(wxyz[1]),
                        "y": float(wxyz[2]),
                        "z": float(wxyz[3]),
                    },
                }

            q_next = np.array(controller.submit_ee_absolute_targets(blended, q_next))
            joint_dict = {
                name: float(val) for name, val in zip(robot.actuated_joint_names, q_next)
            }
            if teleop_loop is not None and teleop_loop.is_running():
                import asyncio

                asyncio.run_coroutine_threadsafe(
                    teleop.publish_joint_state(joint_dict),
                    teleop_loop,
                )
            time.sleep(step_sleep_s)

        time.sleep(hold_s)
        logger.info("Heart gesture finished")
        return q_next
    finally:
        controller.set_mode(original_mode_value)
