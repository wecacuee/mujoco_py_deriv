from argparse import Namespace

import numpy as np
import mujoco_py
from mujoco_py_deriv import MjDerivative

from .mujoco_utils import (mjc_qpos_indices_from_jnt_names,
                           mjc_dof_indices_from_jnt_names)
from .cacheutils import cached_property

TOL = 1e-6


def safe_div(x, y, default=0, tol=TOL):
    return np.where(np.abs(y) < tol, default, x / y)


class MujocoDynamics:
    def __init__(self, mjsim, joints, dt=None, goal_body="fingertip"):
        self.joints = joints
        self.sim = mjsim
        self.model = mjsim.model
        self.data = mjsim.data
        if dt is not None:
            self.model.opt.timestep = dt
        self.goal_body = goal_body
        self.mjderiv = MjDerivative(self.model, self.data, ["qacc"],
                                    ["qfrc_applied", "qvel", "qpos"],
                                    isforward=1)
        self._cache = Namespace()
        self._cache.state = np.empty(self.state_size)
        self._cache.f_x = np.empty((self.state_size, self.state_size))
        self._cache.f_u = np.empty((self.state_size, self.action_size))
        self._cache.deriv = np.empty(self.mjderiv.ext.deriv_shape(),
                                     dtype=np.float64)
        if self.model.nq != self.model.nv:
            raise NotImplementedError("Not implemented for quaternions")

    @cached_property
    def qpos_indices(self):
        return mjc_qpos_indices_from_jnt_names(self.model, self.joints)

    @cached_property
    def dof_indices(self):
        return mjc_dof_indices_from_jnt_names(self.model, self.joints)

    @property
    def state_size(self):
        return len(self.qpos_indices) + len(self.dof_indices)

    @property
    def action_size(self):
        """Action size."""
        return self.model.nu

    @property
    def has_hessians(self):
        """Whether the second order derivatives are available."""
        return False

    def _set_temp_data(self):
        self.sim.data = None

    def _set_data(self, x, u):
        assert not np.isnan(x).any() and not np.isinf(x).any()
        if u is not None:
            assert not np.isnan(u).any() and not np.isinf(x).any()
        qpos = self.data.qpos.reshape(-1).copy()
        qpos[self.qpos_indices] = x[:len(self.qpos_indices)]
        qvel = self.data.qvel.reshape(-1).copy()
        qvel[self.dof_indices] = x[len(self.qpos_indices):]
        self.set_state(qpos, qvel)
        # Only computes end-effector position etc. Is not same as
        self.sim.forward()
        if u is not None:
            self.sim.data.ctrl[:] = u
        # self.sim.step()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, ) and qvel.shape == (
            self.model.nv, )
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)

    def get_state(self):
        state = self._cache.state
        state[:len(self.qpos_indices)] = self.data.qpos[self.qpos_indices]
        state[len(self.qpos_indices):] = self.data.qvel[self.dof_indices]
        return state

    x0 = property(get_state)

    def f(self, x, u, i):
        """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            Next state [state_size].
        """
        self._set_data(x, u)
        self.do_simulation()
        state = self.get_state()
        assert not np.isnan(state).any() and not np.isinf(state).any()
        return state

    def f_x(self, x, u, i):
        """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/dx [state_size, state_size].
        """
        self._set_data(x, u)
        deriv = self.mjderiv.compute(self._cache.deriv)
        assert not np.isnan(deriv).any() and not np.isinf(deriv).any()

        dqacc__dqvel = deriv[0, 1, self.dof_indices, :][:, self.dof_indices]
        dqacc__dqpos = deriv[0, 2, self.dof_indices, :][:, self.dof_indices]
        dqvel__dqpos = safe_div(self.data.qacc[self.dof_indices].reshape(-1, 1),
                                self.data.qvel[self.dof_indices].reshape(1, -1))

        J_x = self._cache.f_x
        J_x[:dqvel__dqpos.shape[0], :dqvel__dqpos.shape[1]] = dqvel__dqpos
        J_x[:dqvel__dqpos.shape[0], dqvel__dqpos.shape[1]:] = np.eye(
            len(self.dof_indices))
        J_x[dqvel__dqpos.shape[0]:, :dqvel__dqpos.shape[1]] = dqacc__dqpos
        J_x[dqvel__dqpos.shape[0]:, dqvel__dqpos.shape[1]:] = dqacc__dqvel
        assert not np.isnan(J_x).any() and not np.isinf(J_x).any()
        return J_x

    def f_u(self, x, u, i):
        """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].
            i: Current time step.

        Returns:
            df/du [state_size, action_size].
        """
        self._set_data(x, u)
        deriv = self.mjderiv.compute(self._cache.deriv)
        assert not np.isnan(deriv).any()

        dqacc__dqfrc_applied = deriv[0, 0, self.dof_indices, :][:, self.dof_indices]
        dqacc__dqvel = deriv[0, 1, self.dof_indices, :][:, self.dof_indices]

        assert (self.model.actuator_dyntype == 0).all()
        assert (self.model.actuator_gaintype == 0).all()
        assert (self.model.actuator_biastype == 0).all()

        nv = len(self.dof_indices)
        dqfrc_applied__dctrl = np.ones(
            (nv, 1)) * self.data.actuator_length.ravel()

        dqacc__dctrl = dqacc__dqfrc_applied.dot(dqfrc_applied__dctrl)
        assert not np.isnan(dqacc__dctrl).any()

        dqvel__dqacc = safe_div(1, dqacc__dqvel)
        dqvel__dctrl = dqvel__dqacc.T.dot(dqacc__dctrl)
        assert not np.isnan(dqvel__dctrl).any()

        J_u = self._cache.f_u
        J_u[:dqvel__dctrl.shape[0], :] = dqvel__dctrl
        J_u[dqvel__dctrl.shape[0]:, :] = dqacc__dctrl
        assert not np.isnan(J_u).any() and not np.isinf(J_u).any()
        return J_u

    def get_body_jac(self, name):
        bodyid = self.model.body_name2id(name)
        bodyxpos_dqpos = self.data.body_jacp[bodyid, :].reshape(-1, 3).T
        return bodyxpos_dqpos

    def dgoal__dqpos(self):
        return self.get_body_jac(self.goal_body)[:, self.qpos_indices]

    def dgoal__dx(self, x, u):
        qpos = self.data.qpos.reshape(-1).copy()
        qpos[self.qpos_indices] = x[:len(self.qpos_indices)]
        qvel = self.data.qvel.reshape(-1).copy()
        qvel[self.dof_indices] = x[len(self.qpos_indices):]
        self._set_data(x, u)
        dgoal__dqpos = self.dgoal__dqpos()
        qacc = self.data.qacc
        dqpos__dqvel = safe_div(self.data.qvel[self.dof_indices].reshape(-1, 1),
                                self.data.qacc[self.dof_indices].reshape(1, -1))
        dgoal__dqvel = dgoal__dqpos.dot(dqpos__dqvel)
        dgoal__dx = np.hstack((dgoal__dqpos, dgoal__dqvel))
        return dgoal__dx

    @property
    def goal_size(self):
        return 3

    def d2goal__d2x(self, x, u):
        return np.zeros((self.goal_size, self.state_size, self.state_size))


    @property
    def dt(self):
        return self.model.opt.timestep

    def do_simulation(self):
        self.sim.step()

    @classmethod
    def augment_state(self, x):
        return x

    @classmethod
    def reduce_state(self, x):
        return x

    def hessian_not_supported(self, *a):
        raise NotImplementedError("Hessians not supported")

    f_xx = hessian_not_supported
    f_ux = hessian_not_supported
    f_uu = hessian_not_supported
