from typing import Any, Optional, TypeVar
import casadi as cs
from csnlp.wrappers.mpc.mpc import Mpc
from csnlp import Nlp, Solution
from vehicle import Vehicle
import numpy as np
from utils.solver_options import solver_options

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

dt = 1

v_max = (w_e_max * r_r * 2 * np.pi) / (z_t[-1] * z_f * 60)
v_min = (w_e_idle * r_r * 2 * np.pi) / (z_t[0] * z_f * 60)
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
a2 = (C_wind * v_max**2 - beta) / (v_max - alpha)
b = beta - alpha * ((C_wind * v_max**2 - beta) / (v_max - alpha))

# fuel cost parameters
p_0 = 0.04918
p_1 = 0.001897
p_2 = 4.5232e-5
gamma = 0.1  # weight for tracking in cost

A = np.zeros((6, 6))
np.fill_diagonal(A, 1)
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)

Q = cs.diag([1, 0.1])


def nonlinear_hybrid_model(x, u, dt, alpha):
    # TODO add docstring
    # u = [T_e, F_b, gear_1, gear_2, gear_3, gear_4, gear_5, gear_6]"
    n = (z_f / r_r) * sum([z_t[i] * u[i + 2] for i in range(6)])
    a = (
        u[0] * n / m
        - C_wind * x[1] ** 2 / m
        - g * mu * np.cos(alpha)
        - g * np.sin(alpha)
        - u[1] / m
    )
    return x + cs.vertcat(x[1], a) * dt


def nonlinear_model(x, u, dt, alpha):
    # TODO add docstring
    a = u / m - C_wind * x[1] ** 2 / m - g * mu * np.cos(alpha) - g * np.sin(alpha)
    return x + cs.vertcat(x[1], a) * dt


class HybridTrackingMpc(Mpc):

    def __init__(
        self,
        prediction_horizon: int,
        optimize_fuel: bool = False,
        convexify_fuel: bool = False,
        convexify_dynamics: bool = False,
    ):
        # TODO add docstring for this whole file
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)

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
        self.constraint("gear_shift_constraint", A @ gear[:, :-1], ">=", gear[:, 1:])
        # self.constraint(
        #     "gear_shift_constraint", A @ cs.horzcat(gear_prev, gear[:, :-1]), ">=", gear
        # )

        w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max
        )
        # comment this section
        if convexify_dynamics:
            self.constraint("pos_dynam", x[0, 1:], "==", x[0, :-1] + x[1, :-1] * dt)

            delta, _ = self.action("delta", 2, discrete=True, lb=0, ub=1)
            self.constraint("delta_constraint", cs.sum1(delta), "==", 1)
            z, _, _ = self.variable("z", (2, prediction_horizon))

            a = (x[1, 1:] - x[1, :-1]) / dt
            F_r = (
                g * mu * m * np.cos(0)
                + g * m * np.sin(0)
                # + C_wind * x[1, :-1] ** 2
                + C_wind * (a1 * z[0, :])
                + C_wind * (a2 * z[1, :] + delta[1, :] * b)
                + m * a
            )
            for i in range(2):  # TODO get ride of loop
                self.constraint(
                    f"drag_gear_{i}_1",
                    z[i, :],
                    "<=",
                    x[1, :-1] - (1 - delta[i, :]) * v_min,
                )
                self.constraint(
                    f"drag_gear_{i}_2",
                    z[i, :],
                    ">=",
                    x[1, :-1] - (1 - delta[i, :]) * v_max,
                )
                self.constraint(f"drag_gear_{i}_3", z[i, :], "<=", delta[i, :] * v_max)
                self.constraint(f"drag_gear_{i}_4", z[i, :], ">=", delta[i, :] * v_min)
            for i in range(6):  # TODO get rid of loop
                n_i = z_f * z_t[i] / r_r
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
            self.set_nonlinear_dynamics(
                lambda x, u: nonlinear_hybrid_model(x, u, dt, 0)
            )

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

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.minimize(
            sum(
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
            self.init_solver(solver_options["gurobi"], solver="gurobi")
        else:
            self.init_solver(solver_options["bonmin"], solver="bonmin")
            # self.init_solver(solver_options["knitro"], solver="knitro")

    def solve(
        self,
        pars: dict,
        vals0: Optional[dict] = None,
    ) -> Solution:
        # TODO add docstring
        vals0 = {
            "w_e": np.full((1, self.prediction_horizon), w_e_idle),
            "T_e": np.full((1, self.prediction_horizon), T_e_idle),
        }  # TODO is this warm start badly biasing
        return self.nlp.solve(pars, vals0)


class HybridTrackingFuelMpcFixedGear(Mpc):

    def __init__(self, prediction_horizon: int, convexify_fuel: bool = False):
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)

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
        n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
        self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))

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

        if convexify_fuel:
            P, _, _ = self.variable("P", (1, prediction_horizon))
            self.constraint(
                "P_lb_1", P, ">=", T_e_idle * w_e + w_e_idle * T_e - w_e_idle * T_e_idle
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

        self.minimize(
            sum(
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
        # TODO add docstring
        gear = pars["gear"]
        if not all(
            np.isclose(np.sum(gear[:, i], axis=0), 1) for i in range(gear.shape[1])
        ):
            raise ValueError("More than one gear selected for a time step.")
        vals0 = {
            "w_e": np.full((1, self.prediction_horizon), w_e_idle),
            "T_e": np.full((1, self.prediction_horizon), T_e_idle),
        }  # TODO is this warm start badly biasing
        return self.nlp.solve(pars, vals0)


class TrackingMpc(Mpc):

    def __init__(self, prediction_horizon: int):
        # TODO add docstring for this whole file
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)

        F_trac_max = self.parameter("F_trac_max", (1, 1))
        F_trac, _ = self.action(
            "F_trac", 1, lb=T_e_idle * z_t[-1] * z_f / r_r - F_b_max
        )  # TODO add torque rate constraint
        self.constraint("traction_force", F_trac, "<=", F_trac_max)

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.set_nonlinear_dynamics(lambda x, u: nonlinear_model(x, u, dt, 0))
        self.minimize(
            sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
        )
        self.init_solver(solver_options["ipopt"], solver="ipopt")
