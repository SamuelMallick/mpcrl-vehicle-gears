from typing import Literal, Optional
from mpcs.mpc import VehicleMPC
import numpy as np
import casadi as cs

from utils.solver_options import solver_options


class NonlinearMPC(VehicleMPC):
    """An MPC controller that uses a nonlinear model of the vehicle dynamics.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    solver : str
        The solver to use for the optimization problem. Options are 'ipopt' (NLP).
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    extra_opts : dict, optional
        Extra options for the solver, by default None.
    """

    def nonlinear_model(self, x: cs.SX, u: cs.SX, dt: float, alpha: float) -> cs.SX:
        """Function for the nonlinear vehicle dynamics x^+ = f(x, u).

        Parameters
        ----------
        x : cs.SX
            State vector [d, v] (m, m/s).
        u : cs.SX
            Input F_trac.
        dt : float
            Time step (s).
        alpha : float
            Road gradient (radians).

        Returns
        -------
        cs.SX
            New state vector [d, v] (m, m/s)."""
        a = (
            u / self.m
            - self.C_wind * x[1] ** 2 / self.m
            - self.g * self.mu * np.cos(alpha)
            - self.g * np.sin(alpha)
        )
        return x + cs.vertcat(x[1], a) * dt

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["ipopt"],
        multi_starts: int = 1,
        extra_opts: Optional[dict] = None,
    ):
        super().__init__(
            prediction_horizon=prediction_horizon,
            solver=solver,
            multi_starts=multi_starts,
        )

        self.F_trac_min = (
            self.T_e_idle * self.z_t[-1] * self.z_f / self.r_r - self.F_b_max
        )

        # explicit velocity constraints in place of engine speed constraints
        self.constraint("v_ub", self.x[1, :], "<=", self.v_max)
        self.constraint("v_lb", self.x[1, :], ">=", self.v_min)

        F_trac_max = self.parameter("F_trac_max", (1, 1))
        F_trac, _ = self.action("F_trac", 1, lb=self.F_trac_min)
        self.constraint("traction_force", F_trac, "<=", F_trac_max)

        self.set_nonlinear_dynamics(lambda x, u: self.nonlinear_model(x, u, self.dt, 0))
        self.minimize(self.tracking_cost)
        opts = solver_options[solver]
        if extra_opts is not None:
            opts[solver].update(extra_opts[solver])
        self.init_solver(opts, solver=solver)
