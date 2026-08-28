# pyright: reportCallIssue=false
import os
import sys
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy

from teleop_xr.ik.robot import BaseRobot, Cost
from teleop_xr import ram


# v2.0 gripper base (preferred) then v1.0 wrist link (test fixtures / old URDFs).
_LEFT_EE_CANDIDATES = ("openarm_left_ee_base_link", "openarm_left_link7")
_RIGHT_EE_CANDIDATES = ("openarm_right_ee_base_link", "openarm_right_link7")


def _resolve_ee_link(link_names: list[str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in link_names:
            return name
    raise ValueError(f"Link {candidates[-1]} not found in URDF")


class OpenArmRobot(BaseRobot):
    """
    OpenArm bimanual robot implementation for IK.

    Loads OpenArm v2.0 from openarm_description (default_bimanual + pinch
    grippers). Older v1.0 URDFs that still use ``*_link7`` as the EE are
    accepted when passed via ``urdf_string``.
    """

    def __init__(self, urdf_string: str | None = None, **kwargs: Any) -> None:
        super().__init__()
        self._robot_preset = str(kwargs.get("robot_preset", "default_bimanual"))
        urdf = self._load_urdf(urdf_string)

        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

        link_names = list(self.robot.links.names)
        self.L_ee: str = _resolve_ee_link(link_names, _LEFT_EE_CANDIDATES)
        self.R_ee: str = _resolve_ee_link(link_names, _RIGHT_EE_CANDIDATES)
        self.L_ee_link_idx: int = self.robot.links.names.index(self.L_ee)
        self.R_ee_link_idx: int = self.robot.links.names.index(self.R_ee)

    def _load_default_urdf(self) -> yourdfpy.URDF:
        repo_url = "https://github.com/enactic/openarm_description.git"
        xacro_path = "assets/robot/openarm_v2.0/urdf/openarm_v20.urdf.xacro"
        xacro_args = {
            "robot_preset": self._robot_preset,
            "use_fake_hardware": "true",
            "collapse_internal_empty_links": "true",
        }

        self.urdf_path = str(
            ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args=xacro_args,
                resolve_packages=True,
                convert_dae_to_glb=True,
            )
        )

        repo_path = ram.get_repo(repo_url)
        self.mesh_path = str(repo_path)

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"OpenArm URDF not found at {self.urdf_path}")

        return yourdfpy.URDF.load(self.urdf_path)

    @property
    @override
    def model_scale(self) -> float:
        return 1.0

    @property
    @override
    def supported_frames(self) -> set[str]:
        return {"left", "right"}

    @property
    @override
    def joint_var_cls(self) -> Any:
        return self.robot.joint_var_cls

    @property
    @override
    def actuated_joint_names(self) -> list[str]:
        return list(self.robot.joints.actuated_names)

    @override
    def forward_kinematics(self, config: jax.Array) -> dict[str, jaxlie.SE3]:
        fk = self.robot.forward_kinematics(config)
        return {
            "left": jaxlie.SE3(fk[self.L_ee_link_idx]),
            "right": jaxlie.SE3(fk[self.R_ee_link_idx]),
        }

    @override
    def get_default_config(self) -> jax.Array:
        joint_names = self.actuated_joint_names
        jnp.zeros(len(joint_names))

        default_pose = {
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": -0.8,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 1.2,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
            "openarm_right_joint1": 0.0,
            "openarm_right_joint2": 0.8,
            "openarm_right_joint3": 0.0,
            "openarm_right_joint4": 1.2,
            "openarm_right_joint5": 0.0,
            "openarm_right_joint6": 0.0,
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint1": 0.0,
            "openarm_left_finger_joint2": 0.0,
            "openarm_right_finger_joint1": 0.0,
            "openarm_right_finger_joint2": 0.0,
        }

        config_list = []
        for name in joint_names:
            config_list.append(default_pose.get(name, 0.0))

        return jnp.array(config_list)

    @override
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        costs = []
        JointVar = self.robot.joint_var_cls

        if q_current is not None:
            costs.append(
                pk.costs.rest_cost(
                    JointVar(0),
                    rest_pose=q_current,
                    weight=5.0,
                )
            )

        costs.append(
            pk.costs.manipulability_cost(
                self.robot,
                JointVar(0),
                jnp.array([self.L_ee_link_idx, self.R_ee_link_idx], dtype=jnp.int32),
                weight=0.01,
            )
        )

        if target_L is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_L,
                    jnp.array(self.L_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        if target_R is not None:
            costs.append(
                pk.costs.pose_cost_analytic_jac(
                    self.robot,
                    JointVar(0),
                    target_R,
                    jnp.array(self.R_ee_link_idx, dtype=jnp.int32),
                    pos_weight=50.0,
                    ori_weight=10.0,
                )
            )

        costs.append(pk.costs.limit_cost(self.robot, JointVar(0), weight=100.0))

        costs.append(
            pk.costs.self_collision_cost(
                self.robot,
                self.robot_coll,
                JointVar(0),
                margin=0.05,
                weight=10.0,
            )
        )

        return costs
