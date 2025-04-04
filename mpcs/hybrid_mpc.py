from typing import Literal
from mpcs.mpc import VehicleMPC
import casadi as cs
import numpy as np


class HybridMPC(VehicleMPC):
    """Base class for MPC controllers that use hybrid models. Includes
    drive-train dynamics such that fuel consumption can be optimized.
    The class is not intended to be used directly, but rather as a base
    class for specific MPC controllers.

    Parameters
    ----------
    rediction_horizon : int
        The length of the prediction horizon.
    solver : str
        The solver to use for the optimization problem. Options are 'ipopt' (NLP),
        'bonmin' (MINLP), 'gurobi' (MIQP), and 'knitro' (NLP).
    optimize_fuel : bool
        Whether to optimize fuel consumption or not.
    convexify_fuel : bool, optional
        Whether to convexify the fuel consumption function or not, by default False,
        and the model is kept in bilinear form. If True, a Mckormick relaxation is
        used.
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    """

    # fuel cost parameters
    p_0 = 0.04918
    p_1 = 0.001897
    p_2 = 4.5232e-5

    # adjacency matrix for gear shift constraints
    A = np.zeros((6, 6))
    np.fill_diagonal(A, 1)
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)

    # weight for tracking in cost. This is a possible incorrect
    # copy of the value in env.py, to allow for an incorrect
    # mpc formulation
    gamma = 0.01

    def nonlinear_hybrid_model(
        self, x: cs.SX, u: cs.SX, dt: float, alpha: float
    ) -> cs.SX:
        """Function for the nonlinear hybrid vehicle dynamics x^+ = f(x, u).

        Parameters
        ----------
        x : cs.SX
            State vector [d, v] (m, m/s).
        u : cs.SX
            Input vector [T_e, F_b, gear_1, gear_2, gear_3, gear_4, gear_5, gear_6].
        dt : float
            Time step (s).
        alpha : float
            Road gradient (radians).

        Returns
        -------
        cs.SX
            New state vector [d, v] (m, m/s).
        """
        n = (self.z_f / self.r_r) * sum([self.z_t[i] * u[i + 2] for i in range(6)])
        a = (
            u[0] * n / self.m
            - self.C_wind * x[1] ** 2 / self.m
            - self.g * self.mu * np.cos(alpha)
            - self.g * np.sin(alpha)
            - u[1] / self.m
        )
        return x + cs.vertcat(x[1], a) * dt

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["ipopt", "bonmin", "gurobi", "knitro"],
        optimize_fuel: bool,
        convexify_fuel: bool = False,
        multi_starts: int = 1,
    ):
        super().__init__(prediction_horizon, solver=solver, multi_starts=multi_starts)
        # engine torque input
        self.T_e, _ = self.action("T_e", 1, lb=self.T_e_idle, ub=self.T_e_max)
        T_e_prev = self.parameter("T_e_prev", (1, 1))
        self.constraint(
            "engine_torque_rate_ub",
            cs.horzcat(T_e_prev, self.T_e[:-1]) - self.T_e,
            "<=",
            self.dT_e_max,
        )
        self.constraint(
            "engine_torque_rate_lb",
            cs.horzcat(T_e_prev, self.T_e[:-1]) - self.T_e,
            ">=",
            -self.dT_e_max,
        )

        # brake force input
        self.F_b, _ = self.action("F_b", 1, lb=0, ub=self.F_b_max)

        # engine speed variable
        self.w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=self.w_e_idle, ub=self.w_e_max
        )
        # w_e_plus is an auxillary variable to check that the engine speed
        # constraints are satisfied at the next time step, such that they
        # are constrained between timesteps aswell
        self.w_e_plus, _, _ = self.variable(
            "w_e_plus", (1, prediction_horizon - 1), lb=self.w_e_idle, ub=self.w_e_max
        )

        self.fuel_consumption = 0
        if optimize_fuel:
            if convexify_fuel:
                # the following two constraints are a McCormick relaxation of the bilinear term P = T_e * w_e
                P, _, _ = self.variable("P", (1, prediction_horizon))
                self.constraint(
                    "P_lb_1",
                    P,
                    ">=",
                    self.T_e_idle * self.w_e
                    + self.w_e_idle * self.T_e
                    - self.w_e_idle * self.T_e_idle,
                )
                self.constraint(
                    "P_lb_2",
                    P,
                    ">=",
                    self.T_e_max * self.w_e
                    + self.w_e_max * self.T_e
                    - self.w_e_max * self.T_e_max,
                )
                self.fuel_consumption += sum(
                    [
                        self.dt * (self.p_0 + self.p_1 * self.w_e[i] + self.p_2 * P[i])
                        for i in range(prediction_horizon)
                    ]
                )
            else:
                self.fuel_consumption += sum(
                    [
                        self.dt
                        * (
                            self.p_0
                            + self.p_1 * self.w_e[i]
                            + self.p_2 * self.w_e[i] * self.T_e[i]
                        )
                        for i in range(prediction_horizon)
                    ]
                )

        # cost of optimization problem
        self.minimize(self.gamma * self.tracking_cost + self.fuel_consumption)
