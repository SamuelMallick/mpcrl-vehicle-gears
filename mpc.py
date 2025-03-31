from typing import Literal, Optional
import casadi as cs
from csnlp.wrappers.mpc.mpc import Mpc
from csnlp import Nlp, Solution
from csnlp.multistart.multistart_nlp import ParallelMultistartNlp
import numpy as np
from utils.solver_options import solver_options

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

dt = 1  # time step (s)

# maximum and minimu velocity: calculated from maximum engine speed: w_e = (z_f * z_t[gear] * 60)/(r_r * 2 * pi)
v_max = (w_e_max * r_r * 2 * np.pi) / (z_t[-1] * z_f * 60)
v_min = (w_e_idle * r_r * 2 * np.pi) / (z_t[0] * z_f * 60)
a_max = 3  # maximum acceleration (deceleration) (m/s^2)

# maximum values of F_r in T_e *n = F_r + F_b
F_r_max = (
    g * mu * m * np.cos(0)
    + g * m * np.sin(0)
    + C_wind * v_max**2
    + m * (v_max - v_min) / dt
)
F_r_min = (
    g * mu * m * np.cos(0)
    + g * m * np.sin(0)
    + C_wind * v_min**2
    + m * (v_min - v_max) / dt
)

# drag approximation parameters
beta = (3 * C_wind * v_max**2) / (16)
alpha = v_max / 2
a1 = beta / alpha
a2 = (13 * C_wind * v_max) / 8
b = (-5 * C_wind * v_max**2) / 8
# a2 = (C_wind * v_max**2 - beta) / (v_max - alpha)
# b = beta - alpha * ((C_wind * v_max**2 - beta) / (v_max - alpha))

# fuel cost parameters
p_0 = 0.04918
p_1 = 0.001897
p_2 = 4.5232e-5

# adjacency matrix for gear shift constraints
A = np.zeros((6, 6))
np.fill_diagonal(A, 1)
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)

# cost matrix for tracking
Q = cs.diag([1, 0.1])
gamma = 0.01  # weight for tracking in cost

d_safe = 10


def nonlinear_hybrid_model(x: cs.SX, u: cs.SX, dt: float, alpha: float) -> cs.SX:
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
    n = (z_f / r_r) * sum([z_t[i] * u[i + 2] for i in range(6)])
    a = (
        u[0] * n / m
        - C_wind * x[1] ** 2 / m
        - g * mu * np.cos(alpha)
        - g * np.sin(alpha)
        - u[1] / m
    )
    return x + cs.vertcat(x[1], a) * dt


def nonlinear_model(x: cs.SX, u: cs.SX, dt: float, alpha: float) -> cs.SX:
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
    a = u / m - C_wind * x[1] ** 2 / m - g * mu * np.cos(alpha) - g * np.sin(alpha)
    return x + cs.vertcat(x[1], a) * dt


class HybridTrackingMpc(Mpc):
    """An mpc controller for controlling the vehicle with the nonlinear hybrid model.
    The controller minimizes a sum of tracking (if optimize_fuel is True) and fuel costs,
    with tracking weighted by gamma. By default the optimization problem is constructed
    as an MINLP, however; if the fuel and dynamics are convexified, the problem is
    an MIQP. Appropriate solvers are initialized based on the convexification settings.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    optimize_fuel : bool, optional
        Whether to optimize fuel consumption, by default False.
    convexify_fuel : bool, optional
        Whether to convexify the fuel cost, using a McCormick relaxation, by default False.
    convexify_dynamics : bool, optional
        Whether to convexify the dynamics, by default False. If true, the quadratic
        friction term is replaced with a piecewise linear function with two segments.
        Further, the bilinear term multiplying the gear by T_e is modelled with mixed
        integer inequalities."""

    def __init__(
        self,
        prediction_horizon: int,
        solver: Literal["bonmin", "gurobi", "knitro"] = "bonmin",
        optimize_fuel: bool = False,
        convexify_fuel: bool = False,
        convexify_dynamics: bool = False,
        multi_starts: int = 1,
    ):
        nlp = ParallelMultistartNlp[cs.SX](sym_type="SX", starts=multi_starts)
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)
        self.constraint("a_ub", x[1, 1:] - x[1, :-1], "<=", a_max * dt)
        self.constraint("a_lb", x[1, 1:] - x[1, :-1], ">=", -a_max * dt)

        # optional constraints for collision with vehicles ahead and behind
        # p_a and p_b are set to arbitrary large and small values to disable
        # the constraints if they are not passed as parameters
        p_a = self.parameter("p_a", (1, prediction_horizon + 1))
        p_b = self.parameter("p_b", (1, prediction_horizon + 1))
        self.constraint("collision_ahead", x[0, :] - p_a, "<=", -d_safe, soft=False)
        self.constraint("collision_behind", x[0, :] - p_b, ">=", d_safe, soft=False)

        T_e, _ = self.action("T_e", 1, lb=T_e_idle, ub=T_e_max)
        T_e_prev = self.parameter("T_e_prev", (1, 1))
        self.constraint(
            "engine_torque_rate_ub",
            cs.horzcat(T_e_prev, T_e[:-1]) - T_e,
            "<=",
            dT_e_max,
        )
        self.constraint(
            "engine_torque_rate_lb",
            cs.horzcat(T_e_prev, T_e[:-1]) - T_e,
            ">=",
            -dT_e_max,
        )

        F_b, _ = self.action("F_b", 1, lb=0, ub=F_b_max)

        gear, _ = self.action("gear", 6, discrete=True, lb=0, ub=1)
        gear_prev = self.parameter("gear_prev", (6, 1))
        self.constraint("gear_constraint", cs.sum1(gear), "==", 1)
        self.constraint(
            "gear_shift_constraint", A @ cs.horzcat(gear_prev, gear[:, :-1]), ">=", gear
        )

        w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max
        )
        # w_e_plus is an auxillary variable to check that the engine speed
        # constraints are satisfied at the next time step, such that they
        # are constrained between timesteps aswell
        w_e_plus, _, _ = self.variable(
            "w_e_plus", (1, prediction_horizon - 1), lb=w_e_idle, ub=w_e_max
        )

        if (
            convexify_dynamics
        ):  # dyanmics are added manually via mixed integer inequalities

            # equality constraint for position dynamics
            self.constraint("pos_dynam", x[0, 1:], "==", x[0, :-1] + x[1, :-1] * dt)

            # delta are binary variables that select region of piecewise affine approximation
            # of the quadratic friction term
            delta, _ = self.action("delta", 2, discrete=True, lb=0, ub=1)
            # constraint such that one region active at each time step
            self.constraint("delta_constraint", cs.sum1(delta), "==", 1)
            # z are auxillary variables z = \delta * v
            z, _, _ = self.variable("z", (2, prediction_horizon))

            a = (x[1, 1:] - x[1, :-1]) / dt
            F_r = (
                g * mu * m * np.cos(0)
                + g * m * np.sin(0)
                + (a1 * z[0, :])
                + (a2 * z[1, :] + delta[1, :] * b)
                + m * a
            )

            # the following four constraint enforce the relation z_i = \delta_i * v
            self.constraint(
                f"z_constraint_1",
                z,
                "<=",
                cs.repmat(x[1, :-1], 2, 1) - (1 - delta) * v_min,
            )
            self.constraint(
                f"z_constraint_2",
                z,
                ">=",
                cs.repmat(x[1, :-1], 2, 1) - (1 - delta) * v_max,
            )
            self.constraint(f"z_constraint_3", z, "<=", delta * v_max)
            self.constraint(f"z_constraint_4", z, ">=", delta * v_min)

            for i in range(6):
                n_i = z_f * z_t[i] / r_r
                # the following two constraints enforce the relation w_e[k] = n[k] * v[k]
                self.constraint(
                    f"engine_speed_gear_{i}_1",
                    w_e + (1 - gear[i, :]) * (n_i * v_max * 60 / (2 * np.pi)),
                    ">=",
                    n_i * x[1, :-1] * 60 / (2 * np.pi),
                )
                self.constraint(
                    f"engine_speed_gear_{i}_2",
                    w_e + (1 - gear[i, :]) * (-w_e_max),
                    "<=",
                    n_i * x[1, :-1] * 60 / (2 * np.pi),
                )
                if i < 5:
                    # the following two constraints enforce the relation w_e_plus[k+1] = n[k] * v[k+1]
                    self.constraint(
                        f"engine_speed_plus_gear_{i}_1",
                        w_e_plus
                        + (1 - gear[i, :-1]) * (n_i * v_max * 60 / (2 * np.pi)),
                        ">=",
                        n_i * x[1, 1:-1] * 60 / (2 * np.pi),
                    )
                    self.constraint(
                        f"engine_speed_plus_gear_{i}_2",
                        w_e_plus + (1 - gear[i, :-1]) * (-w_e_max),
                        "<=",
                        n_i * x[1, 1:-1] * 60 / (2 * np.pi),
                    )
                # the following two constraints enforce the relation T_e * n[i] = F_r + F_b
                self.constraint(
                    f"dynam_gear_{i}_1",
                    T_e * n_i + (1 - gear[i, :]) * (F_r_max + F_b_max),
                    ">=",
                    F_r + F_b,
                )
                self.constraint(
                    f"dynam_gear_{i}_2",
                    T_e * n_i + (1 - gear[i, :]) * (F_r_min - T_e_max * n_i),
                    "<=",
                    F_r + F_b,
                )
        else:
            n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
            self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))
            self.constraint(
                "engine_speed_plus",
                w_e_plus,
                "==",
                x[1, 1:-1] * n[:, :-1] * 60 / (2 * np.pi),
            )
            self.set_nonlinear_dynamics(
                lambda x, u: nonlinear_hybrid_model(x, u, dt, 0)
            )

        if optimize_fuel:
            if convexify_fuel:
                # the following two constraints are a McCormick relaxation of the bilinear term P = T_e * w_e
                P, _, _ = self.variable("P", (1, prediction_horizon))
                self.constraint(
                    "P_lb_1",
                    P,
                    ">=",
                    T_e_idle * w_e + w_e_idle * T_e - w_e_idle * T_e_idle,
                )
                self.constraint(
                    "P_lb_2", P, ">=", T_e_max * w_e + w_e_max * T_e - w_e_max * T_e_max
                )
                fuel_cost = sum(
                    [
                        dt * (p_0 + p_1 * w_e[i] + p_2 * P[i])
                        for i in range(prediction_horizon)
                    ]
                )
            else:
                fuel_cost = sum(
                    [
                        dt * (p_0 + p_1 * w_e[i] + p_2 * w_e[i] * T_e[i])
                        for i in range(prediction_horizon)
                    ]
                )
        else:
            fuel_cost = 0

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.minimize(
            gamma
            * sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
            + fuel_cost
        )
        if (
            convexify_dynamics
            and not optimize_fuel
            or convexify_dynamics
            and optimize_fuel
            and convexify_fuel
        ):
            if solver != "gurobi":
                Warning("Using gurobi solver for convexified problem, solver changed.")
            self.init_solver(solver_options["gurobi"], solver="gurobi")
        else:
            if solver == "gurobi":
                raise ValueError(
                    "Gurobi solver can only be used for convexified problems."
                )
            self.init_solver(solver_options[solver], solver=solver)

    def solve(
        self,
        pars: dict,
        vals0: Optional[dict] = None,
    ) -> Solution:
        if "p_a" not in pars:
            pars["p_a"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] + 1e6
            )
        if "p_b" not in pars:
            pars["p_b"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] - 1e6
            )
        return self.nlp.solve_multi(pars, vals0)


class HybridTrackingFuelMpcFixedGear(Mpc):
    """An mpc controller for controlling the vehicle with the nonlinear hybrid model.
    The controller minimizes a sum of tracking (if optimize_fuel is True) and fuel costs,
    with tracking weighted by gamma. The gears are assumed to be fixed, and must be
    provided as parameters to the solve method. The optimization probem is constructed
    as an NLP.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    optimize_fuel : bool, optional
        Whether to optimize fuel consumption, by default False.
    convexify_fuel : bool, optional
        Whether to convexify the fuel cost, using a McCormick relaxation, by default False.
    """

    def __init__(
        self,
        prediction_horizon: int,
        optimize_fuel: bool,
        convexify_fuel: bool = False,
        multi_starts: int = 1,
    ):
        nlp = ParallelMultistartNlp[cs.SX](sym_type="SX", starts=multi_starts)
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)
        self.constraint("a_ub", x[1, 1:] - x[1, :-1], "<=", a_max * dt)
        self.constraint("a_lb", x[1, 1:] - x[1, :-1], ">=", -a_max * dt)

        # optional constraints for collision with vehicles ahead and behind
        # p_a and p_b are set to arbitrary large and small values to disable
        # the constraints if they are not passed as parameters
        p_a = self.parameter("p_a", (1, prediction_horizon + 1))
        p_b = self.parameter("p_b", (1, prediction_horizon + 1))
        self.constraint("collision_ahead", x[0, :] - p_a, "<=", -d_safe, soft=False)
        self.constraint("collision_behind", x[0, :] - p_b, ">=", d_safe, soft=False)

        T_e, _ = self.action("T_e", 1, lb=T_e_idle, ub=T_e_max)
        T_e_prev = self.parameter("T_e_prev", (1, 1))
        self.constraint(
            "engine_torque_rate_ub",
            cs.horzcat(T_e_prev, T_e[:-1]) - T_e,
            "<=",
            dT_e_max,
        )
        self.constraint(
            "engine_torque_rate_lb",
            cs.horzcat(T_e_prev, T_e[:-1]) - T_e,
            ">=",
            -dT_e_max,
        )

        F_b, _ = self.action("F_b", 1, lb=0, ub=F_b_max)

        gear = self.parameter("gear", (6, prediction_horizon))

        w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max
        )
        w_e_plus, _, _ = self.variable(
            "w_e_plus", (1, prediction_horizon - 1), lb=w_e_idle, ub=w_e_max
        )
        n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
        self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))
        self.constraint(
            "engine_speed_plus", w_e_plus, "==", x[1, 1:-1] * n[:-1] * 60 / (2 * np.pi)
        )

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))

        X_next = []
        for k in range(prediction_horizon):
            X_next.append(
                nonlinear_hybrid_model(
                    x[:, k], cs.vertcat(T_e[k], F_b[k], gear[:, k]), dt, 0
                )
            )
        X_next = cs.horzcat(*X_next)
        self.constraint("dynamics", x[:, 1:], "==", X_next)

        if optimize_fuel:
            if convexify_fuel:
                P, _, _ = self.variable("P", (1, prediction_horizon))
                self.constraint(
                    "P_lb_1",
                    P,
                    ">=",
                    T_e_idle * w_e + w_e_idle * T_e - w_e_idle * T_e_idle,
                )
                self.constraint(
                    "P_lb_2", P, ">=", T_e_max * w_e + w_e_max * T_e - w_e_max * T_e_max
                )
                fuel_cost = sum(
                    [
                        dt * (p_0 + p_1 * w_e[i] + p_2 * P[i])
                        for i in range(prediction_horizon)
                    ]
                )
            else:
                fuel_cost = sum(
                    [
                        dt * (p_0 + p_1 * w_e[i] + p_2 * w_e[i] * T_e[i])
                        for i in range(prediction_horizon)
                    ]
                )
        else:
            fuel_cost = 0

        self.minimize(
            gamma
            * sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
            + fuel_cost
        )
        self.init_solver(solver_options["ipopt"], solver="ipopt")

    def solve(
        self,
        pars: dict,
        vals0: Optional[dict] = None,
    ) -> Solution:
        gear = pars["gear"]
        if not all(
            np.isclose(np.sum(gear[:, i], axis=0), 1) for i in range(gear.shape[1])
        ):
            raise ValueError("More than one gear selected for a time step.")
        if "p_a" not in pars:
            pars["p_a"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] + 1e6
            )
        if "p_b" not in pars:
            pars["p_b"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] - 1e6
            )
        return self.nlp.solve_multi(pars, vals0)


class TrackingMpc(Mpc):
    """An mpc controller for controlling the vehicle with the nonlinear model. Fuel
    consumption can not be optimized, as the engine dynamics are not modelled.
    The optimization probem is constructed as an NLP.

    Parameters
    ----------
    prediction_horizon : int
        The length of the prediction horizon.
    """

    F_trac_min = T_e_idle * z_t[-1] * z_f / r_r - F_b_max

    def __init__(self, prediction_horizon: int, multi_starts: int = 1):
        nlp = ParallelMultistartNlp[cs.SX](sym_type="SX", starts=multi_starts)
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)
        self.constraint("a_ub", x[1, 1:] - x[1, :-1], "<=", a_max * dt)
        self.constraint("a_lb", x[1, 1:] - x[1, :-1], ">=", -a_max * dt)
        self.constraint("v_ub", x[1, :], "<=", v_max)
        self.constraint("v_lb", x[1, :], ">=", v_min)

        # optional constraints for collision with vehicles ahead and behind
        # p_a and p_b are set to arbitrary large and small values to disable
        # the constraints if they are not passed as parameters
        p_a = self.parameter("p_a", (1, prediction_horizon + 1))
        p_b = self.parameter("p_b", (1, prediction_horizon + 1))
        self.constraint("collision_ahead", x[0, :] - p_a, "<=", -d_safe, soft=False)
        self.constraint("collision_behind", x[0, :] - p_b, ">=", d_safe, soft=False)

        F_trac_max = self.parameter("F_trac_max", (1, 1))
        F_trac, _ = self.action("F_trac", 1, lb=self.F_trac_min)
        self.constraint("traction_force", F_trac, "<=", F_trac_max)

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.set_nonlinear_dynamics(lambda x, u: nonlinear_model(x, u, dt, 0))
        self.minimize(
            gamma
            * sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
        )
        self.init_solver(solver_options["ipopt"], solver="ipopt")

    def solve(
        self,
        pars: dict,
        vals0: Optional[dict] = None,
    ) -> Solution:
        if "p_a" not in pars:
            pars["p_a"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] + 1e6
            )
        if "p_b" not in pars:
            pars["p_b"] = np.full(
                (1, self.prediction_horizon + 1), pars["x_0"][0] - 1e6
            )
        return self.nlp.solve_multi(pars, vals0)
