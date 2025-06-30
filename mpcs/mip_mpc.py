from mpcs.hybrid_mpc import HybridMPC
from typing import Literal, Optional
import casadi as cs
import numpy as np
import copy

from utils.solver_options import solver_options


class MIPMPC(HybridMPC):
    """An MPC controller that solves a mixed-integer optimization problem
    to optimize over the gear-shift schedule, and the enginer torque and
    brake force.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    solver : str
        The solver to use for the optimization problem. Options are
        'bonmin' (MINLP), 'gurobi' (MIQP), and 'knitro' (NLP).
    optimize_fuel : bool
        Whether to optimize fuel consumption or not.
    convexify_fuel : bool, optional
        Whether to convexify the fuel consumption function or not, by default False,
        and the model is kept in bilinear form. If True, a Mckormick relaxation is
        used.
    convexify_dynamics : bool, optional
        Whether to convexify the dynamics, by default False. If true, the quadratic
        friction term is replaced with a piecewise linear function with two segments.
        Further, the bilinear term multiplying the gear by T_e is modelled with mixed
        integer inequalities. If True the optimization problem is an MIQP, and the
        solver gurobi can be used. Otherwise, the optimization problem is an MINLP,
        and the solvers bonmin or knitro must be used.
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    extra_opts : dict, optional
        Extra options for the solver, by default None.
    """

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["bonmin", "gurobi", "knitro"],
        optimize_fuel: bool,
        convexify_fuel: bool = False,
        convexify_dynamics: bool = False,
        multi_starts: int = 1,
        extra_opts: Optional[dict] = None,
    ):
        super().__init__(
            prediction_horizon=prediction_horizon,
            solver=solver,
            optimize_fuel=optimize_fuel,
            convexify_fuel=convexify_fuel,
            multi_starts=multi_starts,
        )

        # drag parameters for PWA approximation
        beta = (3 * self.C_wind * self.v_max**2) / (16)
        alpha = self.v_max / 2
        a1 = beta / alpha
        a2 = (13 * self.C_wind * self.v_max) / 8
        b = (-5 * self.C_wind * self.v_max**2) / 8

        # maximum values of F_r in T_e *n = F_r + F_b
        F_r_max = (
            self.g * self.mu * self.m * np.cos(0)
            + self.g * self.m * np.sin(0)
            + self.C_wind * self.v_max**2
            + self.m * (self.v_max - self.v_min) / self.dt
        )
        F_r_min = (
            self.g * self.mu * self.m * np.cos(0)
            + self.g * self.m * np.sin(0)
            + self.C_wind * self.v_min**2
            + self.m * (self.v_min - self.v_max) / self.dt
        )

        # gear as discrete binary action
        gear, _ = self.action("gear", 6, discrete=True, lb=0, ub=1)
        gear_prev = self.parameter("gear_prev", (6, 1))
        # restrict one gear to be active at a time
        self.constraint("gear_constraint", cs.sum1(gear), "==", 1)
        # restrict gear shifts to adjacent gears
        self.constraint(
            "gear_shift_constraint",
            self.A @ cs.horzcat(gear_prev, gear[:, :-1]),
            ">=",
            gear,
        )

        # set dynamics
        if (
            convexify_dynamics
        ):  # dyanmics are added manually via mixed integer inequalities

            # equality constraint for position dynamics
            self.constraint(
                "pos_dynam",
                self.x[0, 1:],
                "==",
                self.x[0, :-1] + self.x[1, :-1] * self.dt,
            )

            # delta are binary variables that select region of piecewise affine approximation
            # of the quadratic friction term
            delta, _ = self.action("delta", 2, discrete=True, lb=0, ub=1)
            # constraint such that one region active at each time step
            self.constraint("delta_constraint", cs.sum1(delta), "==", 1)
            # z are auxillary variables z = \delta * v
            z, _, _ = self.variable("z", (2, prediction_horizon))

            a = (self.x[1, 1:] - self.x[1, :-1]) / self.dt
            F_r = (
                self.g * self.mu * self.m * np.cos(0)
                + self.g * self.m * np.sin(0)
                + (a1 * z[0, :])
                + (a2 * z[1, :] + delta[1, :] * b)
                + self.m * a
            )

            # the following four constraint enforce the relation z_i = \delta_i * v
            self.constraint(
                f"z_constraint_1",
                z,
                "<=",
                cs.repmat(self.x[1, :-1], 2, 1) - (1 - delta) * self.v_min,
            )
            self.constraint(
                f"z_constraint_2",
                z,
                ">=",
                cs.repmat(self.x[1, :-1], 2, 1) - (1 - delta) * self.v_max,
            )
            self.constraint(f"z_constraint_3", z, "<=", delta * self.v_max)
            self.constraint(f"z_constraint_4", z, ">=", delta * self.v_min)

            for i in range(6):
                n_i = self.z_f * self.z_t[i] / self.r_r
                # the following two constraints enforce the relation w_e[k] = n[k] * v[k]
                self.constraint(
                    f"engine_speed_gear_{i}_1",
                    self.w_e + (1 - gear[i, :]) * (n_i * self.v_max * 60 / (2 * np.pi)),
                    ">=",
                    n_i * self.x[1, :-1] * 60 / (2 * np.pi),
                )
                self.constraint(
                    f"engine_speed_gear_{i}_2",
                    self.w_e + (1 - gear[i, :]) * (-self.w_e_max),
                    "<=",
                    n_i * self.x[1, :-1] * 60 / (2 * np.pi),
                )
                if i < 5:
                    # the following two constraints enforce the relation w_e_plus[k+1] = n[k] * v[k+1]
                    self.constraint(
                        f"engine_speed_plus_gear_{i}_1",
                        self.w_e_plus
                        + (1 - gear[i, :-1]) * (n_i * self.v_max * 60 / (2 * np.pi)),
                        ">=",
                        n_i * self.x[1, 1:-1] * 60 / (2 * np.pi),
                    )
                    self.constraint(
                        f"engine_speed_plus_gear_{i}_2",
                        self.w_e_plus + (1 - gear[i, :-1]) * (-self.w_e_max),
                        "<=",
                        n_i * self.x[1, 1:-1] * 60 / (2 * np.pi),
                    )
                # the following two constraints enforce the relation T_e * n[i] = F_r + F_b
                self.constraint(
                    f"dynam_gear_{i}_1",
                    self.T_e * n_i + (1 - gear[i, :]) * (F_r_max + self.F_b_max),
                    ">=",
                    F_r + self.F_b,
                )
                self.constraint(
                    f"dynam_gear_{i}_2",
                    self.T_e * n_i + (1 - gear[i, :]) * (F_r_min - self.T_e_max * n_i),
                    "<=",
                    F_r + self.F_b,
                )
        else:
            n = (self.z_f / self.r_r) * sum(
                [self.z_t[i] * gear[i, :] for i in range(6)]
            )
            self.constraint(
                "engine_speed", self.w_e, "==", self.x[1, :-1] * n * 60 / (2 * np.pi)
            )
            self.constraint(
                "engine_speed_plus",
                self.w_e_plus,
                "==",
                self.x[1, 1:-1] * n[:, :-1] * 60 / (2 * np.pi),
            )
            self.set_nonlinear_dynamics(
                lambda x, u: self.nonlinear_hybrid_model(x, u, self.dt, 0)
            )

        opts = copy.deepcopy(solver_options[solver])  # deepcopy to modify dict locally
        if extra_opts is not None:
            opts[solver].update(extra_opts[solver])
        self.init_solver(opts, solver=solver)

    def solve(self, pars, vals0=None):
        if isinstance(pars, dict):
            pars = [pars]
        return super().solve(pars, vals0)
