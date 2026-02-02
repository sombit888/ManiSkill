import numpy as np
import sapien
import torch

from typing import List, Optional, Union

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import common, sapien_utils



# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    ee_link = "ee_gripper_link"
    base_link = 'base_link'
    # Default drift values (can be overridden via constructor)
    default_arm_drift = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 arm joints
    default_gripper_drift = [0.0, 0.0]  # 2 gripper joints

    def __init__(
        self,
        *args,
        arm_drift: Optional[Union[float, List[float]]] = None,
        gripper_drift: Optional[Union[float, List[float]]] = None,
        **kwargs
    ):
        """
        Initialize WidowX250S robot.

        Args:
            arm_drift: Drift values for 6 arm joints [waist, shoulder, elbow, foreaoobsrm_roll, wrist_angle, wrist_rotate].
                       If None, uses default (no drift). Can be a single float for all joints or a list of 6 floats.
            gripper_drift: Drift values for 2 gripper joints [left_finger, right_finger].
                          If None, uses default (no drift). Can be a single float for both or a list of 2 floats.
        """
        # Set drift values before calling parent __init__ (which calls _controller_configs)
        if arm_drift is None:
            self.arm_drift = self.default_arm_drift
        elif isinstance(arm_drift, (int, float)):
            self.arm_drift = [arm_drift] * 6
        else:
            self.arm_drift = list(arm_drift)

        if gripper_drift is None:
            self.gripper_drift = self.default_gripper_drift
        elif isinstance(gripper_drift, (int, float)):
            self.gripper_drift = [gripper_drift] * 2
        else:
            self.gripper_drift = list(gripper_drift)

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # Combine arm and gripper drift values
        all_joint_names = self.arm_joint_names + self.gripper_joint_names
        all_drift = self.arm_drift + self.gripper_drift

        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                joint_names=all_joint_names,
                lower=None,
                upper=None,
                stiffness=100,
                damping=10,
                normalize_action=False,
                drift=all_drift,
            ),
            pd_joint_delta_pos=PDJointPosControllerConfig(
                joint_names=all_joint_names,
                lower=-0.1,
                upper=0.1,
                stiffness=100,
                damping=10,
                normalize_action=True,
                use_delta=True,
                drift=all_drift,
            ),
        )

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link
        )
        self.base_link_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.base_link
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
    @property
    def base_pose(self):
        return self.base_link_tcp.pose
    @property
    def ee_pose(self):
        return self.tcp.pose
    @property
    def gripper_closedness(self):
        qpos = self.robot.get_qpos()          # [B, 8] or [8] for single robot
        qlim = self.robot.get_qlimits()       # [B, 8, 2] or [8, 2]

        # Ensure tensors
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos, device='cuda')
        if not isinstance(qlim, torch.Tensor):
            qlim = torch.as_tensor(qlim, device='cuda')

        # If single robot, unsqueeze to make batch dim
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)       # [1, 8]
        if qlim.ndim == 2:
            qlim = qlim.unsqueeze(0)       # [1, 8, 2]

        # Extract last 2 joints (fingers)
        finger_qpos = qpos[:, -2:]           # [B, 2]
        finger_qlim = qlim[:, -2:, :]        # [B, 2, 2]

        # Compute closedness
        closedness = (finger_qlim[:, :, 1] - finger_qpos) / (finger_qlim[:, :, 1] - finger_qlim[:, :, 0])
        closedness = torch.mean(closedness, dim=1)
        closedness = torch.clamp(closedness, min=0.0)

        # If single robot, return scalar
        if closedness.shape[0] == 1:
            return closedness[0]
        return closedness