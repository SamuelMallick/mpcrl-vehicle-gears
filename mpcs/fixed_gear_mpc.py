from typing import Literal, Optional

import casadi as cs
import numpy as np
from csnlp import Solution

from mpcs.hybrid_mpc import HybridMPC
from utils.solver_options import solver_options


class FixedGearMPC(HybridMPC):
    """An MPC controller that is passed a fixed gear schedule, and optimizes
    an NLP over the engine torque and brake force.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    solver : str
        The solver to use for the optimization problem. Options are 'ipopt' (NLP).
    optimize_fuel : bool
        Whether to optimize fuel consumption or not.
    convexify_fuel : bool, optional
        Whether to convexify the fuel consumption function or not, by default False,
        and the model is kept in bilinear form. If True, a Mckormick relaxation is
        used.
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    extra_opts : dict, optional
        Extra options for the solver, by default None.
    """

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["ipopt"],
        optimize_fuel: bool,
        convexify_fuel: bool = False,
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

        # gear as a fixed parameter
        gear = self.parameter("gear", (6, prediction_horizon))

        n = (self.z_f / self.r_r) * sum([self.z_t[i] * gear[i, :] for i in range(6)])
        self.constraint(
            "engine_speed", self.w_e, "==", self.x[1, :-1] * n * 60 / (2 * np.pi)
        )
        self.constraint(
            "engine_speed_plus",
            self.w_e_plus,
            "==",
            self.x[1, 1:-1] * n[:-1] * 60 / (2 * np.pi),
        )

        # set dynamics
        X_next = []
        for k in range(prediction_horizon):
            X_next.append(
                self.nonlinear_hybrid_model(
                    self.x[:, k],
                    cs.vertcat(self.T_e[k], self.F_b[k], gear[:, k]),
                    self.dt,
                    0,
                )
            )
        X_next = cs.horzcat(*X_next)
        self.constraint("dynamics", self.x[:, 1:], "==", X_next)

        opts = solver_options[solver]
        if extra_opts is not None:
            opts[solver].update(extra_opts[solver])
        self.init_solver(opts, solver=solver)

    def solve(
        self,
        pars: dict | list[dict],
        vals0: Optional[dict] = None,
    ) -> tuple[Solution, dict]:
        if isinstance(pars, dict):
            pars = [pars]
        if any("gear" not in p for p in pars):
            raise ValueError("Fixed gear schedule must be provided.")
        if any("gear_prev" not in p for p in pars):
            raise ValueError("Previous gear position must be provided.")
        gear = [p["gear"] for p in pars]
        gear_prev = [p["gear_prev"] for p in pars]
        if not all(
            all(np.isclose(np.sum(g[:, i], axis=0), 1) for i in range(g.shape[1]))
            for g in gear
        ):
            raise ValueError("More than one gear selected for a time step.")
        if not all(
            np.all(self.A @ cs.horzcat(gp, g[:, :-1]) >= g)
            for gp, g in zip(gear_prev, gear)
        ):
            pass
            # print("Warning: Gear-shift schedule skipping gears.")
            # raise ValueError("Gear-shift schedule skipping gears.")
        return super().solve(pars, vals0=vals0)
