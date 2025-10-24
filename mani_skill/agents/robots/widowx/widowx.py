import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


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