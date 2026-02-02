# TODO(jigu): Move to sapien_utils.py

from typing import Sequence
import torch 


import numpy as np
import sapien
import sapien.physx as physx
from gymnasium import spaces

from mani_skill.utils import common
from mani_skill.utils.structs.articulation import Articulation


def get_active_joint_indices(articulation: Articulation, joint_names: Sequence[str]):
    """get the indices of the provided joint names from the Articulation's list of active joints"""
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return common.to_tensor(joint_indices).int()


def get_joints_by_names(articulation: Articulation, joint_names: Sequence[str]):
    """Gets the Joint objects by name in the Articulation's list of active joints"""
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]


def flatten_action_spaces(action_spaces: dict[str, spaces.Space]):
    """Flat multiple Box action spaces into a single Box space."""
    action_dims = []
    low = []
    high = []
    action_mapping = dict()
    offset = 0

    for action_name, action_space in action_spaces.items():
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, (action_name, action_space)
        else:
            raise TypeError(action_space)

        action_dim = action_space.shape[0]
        action_dims.append(action_dim)
        low.append(action_space.low)
        high.append(action_space.high)
        action_mapping[action_name] = (offset, offset + action_dim)
        offset += action_dim

    flat_action_space = spaces.Box(
        low=np.hstack(low),
        high=np.hstack(high),
        shape=[sum(action_dims)],
        dtype=np.float32,
    )

    return flat_action_space, action_mapping
def mat2quat_torch(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to quaternion (w, x, y, z) in PyTorch.
    R: shape [3,3] or [B,3,3]
    Returns: shape [4] or [B,4], same device as R
    """
    # use torch.linalg.eigvals/eigvecs or stable formula
    # Here’s a numerically stable method:
    # Reference: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py

    m = R
    if m.ndim == 2:  # single rotation
        qw = torch.sqrt(1 + m[0,0] + m[1,1] + m[2,2]) / 2
        qx = (m[2,1] - m[1,2]) / (4*qw)
        qy = (m[0,2] - m[2,0]) / (4*qw)
        qz = (m[1,0] - m[0,1]) / (4*qw)
        return torch.stack([qw, qx, qy, qz])
    elif m.ndim == 3:  # batched rotation [B,3,3]
        qw = torch.sqrt(1 + m[:,0,0] + m[:,1,1] + m[:,2,2]) / 2
        qx = (m[:,2,1] - m[:,1,2]) / (4*qw)
        qy = (m[:,0,2] - m[:,2,0]) / (4*qw)
        qz = (m[:,1,0] - m[:,0,1]) / (4*qw)
        return torch.stack([qw, qx, qy, qz], dim=1)
    else:
        raise ValueError("Invalid rotation matrix shape.")