import numpy as np
import mujoco_py as mj

def mjc_addr_to_indices(addr):
    indices = (np.arange(*addr)
               if isinstance(addr, tuple)
               else np.arange(addr, addr+1))
    return indices

def mjc_qpos_indices_from_jnt_names(model, joints):
    return np.hstack([
        mjc_addr_to_indices(model.get_joint_qpos_addr(j))
        for j in joints])


def mjc_dof_indices_from_jnt_names(model, joints):
    return np.hstack([
        mjc_addr_to_indices(model.get_joint_qvel_addr(j))
        for j in joints])
