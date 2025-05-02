from typing import Literal, Optional
import casadi as cs
from csnlp.wrappers.mpc.mpc import Mpc
from csnlp import Solution
from csnlp.multistart.multistart_nlp import ParallelMultistartNlp
import numpy as np
from utils.solver_options import solver_options


class VehicleMPC(Mpc):
    """Base class for MPC controllers for the vehicles. The class includes
    common components and parameters, e.g., states, state constraints, and
    vehicle dynamics coefficients. The class is not intended to be used directly,
    but rather as a base class for specific MPC controllers.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    solver : str
        The solver to use for the optimization problem. Options are 'ipopt' (NLP),
        'bonmin' (MINLP), 'gurobi' (MIQP), and 'knitro' (NLP).
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    max_time : float, optional
        The maximum time to solve the optimization problem, by default None.
    """

    # ---------------------- Vehicle parameters ----------------------
    # these parameters are (identical or not) copies of those in vehicle.py, allowing
    # for the MPC controllers to be defined with incorrect parameters

    m = 1500  # mass of the vehicle (kg)
    C_wind = 0.4071  # wind resistance coefficient
    mu = 0.015  # rolling resistance coefficient
    g = 9.81  # gravitational acceleration (m/s^2)
    r_r = 0.3554  # wheel radius (m)
    z_f = 3.39  # final drive ratio
    z_t = [4.484, 2.872, 1.842, 1.414, 1.000, 0.742]  # gear ratios

    w_e_max = 3000  # maximum engine speed (rpm)
    T_e_max = 300  # maximum engine torque (Nm)
    dT_e_max = 100  # maximum engine torque rate (Nm/s)
    T_e_idle = 15  # engine idle torque (Nm)
    w_e_idle = 900  # engine idle speed (rpm)
    F_b_max = 9000  # maximum braking force (N)

    # maximum and minimum velocity: calculated from maximum engine speed: w_e = (z_f * z_t[gear] * 60)/(r_r * 2 * pi)
    v_max = (w_e_max * r_r * 2 * np.pi) / (z_t[-1] * z_f * 60)
    v_min = (w_e_idle * r_r * 2 * np.pi) / (z_t[0] * z_f * 60)
    a_max = 3  # maximum acceleration (deceleration) (m/s^2)

    dt = 1  # time step (s)

    Q = cs.diag([1, 0.1])  # cost matrix for tracking

    d_safe = 10  # safe distance to other vehicles (m)

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["ipopt", "bonmin", "gurobi", "knitro"],
        multi_starts: int = 1,
        max_time: Optional[float] = None,
    ):
        nlp = ParallelMultistartNlp[cs.SX](sym_type="SX", starts=multi_starts)
        super().__init__(nlp, prediction_horizon)

        self.x, _ = self.state("x", 2)
        # acceleration constraints
        self.constraint(
            "a_ub", self.x[1, 1:] - self.x[1, :-1], "<=", self.a_max * self.dt
        )
        self.constraint(
            "a_lb", self.x[1, 1:] - self.x[1, :-1], ">=", -self.a_max * self.dt
        )

        # optional constraints for collision with vehicles ahead and behind
        # p_a and p_b are set to arbitrary large and small values in solve
        # method to disable the constraints if they are not passed as parameters
        p_a = self.parameter("p_a", (1, prediction_horizon + 1))
        p_b = self.parameter("p_b", (1, prediction_horizon + 1))
        # self.constraint(
        #     "collision_ahead", self.x[0, :] - p_a, "<=", -self.d_safe, soft=False
        # )
        # self.constraint(
        #     "collision_behind", self.x[0, :] - p_b, ">=", self.d_safe, soft=False
        # )
        _, _, s_a = self.constraint(
            "collision_ahead", self.x[0, :] - p_a, "<=", -self.d_safe, soft=True
        )
        _, _, s_b = self.constraint(
            "collision_behind", self.x[0, :] - p_b, ">=", self.d_safe, soft=True
        )

        # reference trajectory to track
        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.tracking_cost = sum(
            [
                cs.mtimes(
                    [(self.x[:, i] - x_ref[:, i]).T, self.Q, self.x[:, i] - x_ref[:, i]]
                )
                for i in range(prediction_horizon + 1)
            ]
        ) + 1e3 * (cs.sum2(s_a) + cs.sum2(s_b))

    def solve(
        self,
        pars: dict | list[dict],
        vals0: Optional[dict] = None,
    ) -> tuple[Solution, dict]:
        for p in pars:
            if "p_a" not in p:
                p["p_a"] = np.full((1, self.prediction_horizon + 1), p["x_0"][0] + 1e6)
            if "p_b" not in p:
                p["p_b"] = np.full((1, self.prediction_horizon + 1), p["x_0"][0] - 1e6)
        # match the num of vals0 and pars
        n = len(vals0) if vals0 is not None else 1
        vals0 = vals0 * len(pars) if vals0 is not None else None
        pars = [p for p in pars for _ in range(n)]

        sols = self.nlp.solve_multi(pars, vals0, return_all_sols=True)
        best_indx = np.argmin([s.f for s in sols])
        best_sol = sols[best_indx]
        longest_time = max([s.stats["t_wall_total"] for s in sols])
        best_sol.stats["t_wall_total"] = longest_time
        return best_sol, {
            "best_indx": best_indx,
            "infeas": [not s.success for s in sols],
        }
