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

p_0 = 0.04918
p_1 = 0.001897
p_2 = 4.5232e-5
gamma = 0.1  # weight for tracking in cost

dt = 1

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

    def __init__(self, prediction_horizon: int):
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
        self.constraint("gear_constraint", cs.sum1(gear), "==", 1)

        w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max
        )
        n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
        self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.set_nonlinear_dynamics(lambda x, u: nonlinear_hybrid_model(x, u, dt, 0))
        self.minimize(
            sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
        )
        self.init_solver(solver_options["bonmin"], solver="bonmin")

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


class HybridTrackingFuelMpc(Mpc):

    def __init__(self, prediction_horizon: int, mckormick_fuel: bool = False):
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
        self.constraint("gear_constraint", cs.sum1(gear), "==", 1)

        w_e, _, _ = self.variable(
            "w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max
        )
        n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
        self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))

        x_ref = self.parameter("x_ref", (2, prediction_horizon + 1))
        self.set_nonlinear_dynamics(lambda x, u: nonlinear_hybrid_model(x, u, dt, 0))

        if mckormick_fuel:
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
            gamma
            * sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
            + fuel_cost
        )
        self.init_solver(solver_options["bonmin"], solver="bonmin")


class HybridTrackingFuelMpcFixedGear(Mpc):

    def __init__(self, prediction_horizon: int, mckormick_fuel: bool = False):
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

        if mckormick_fuel:
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
        if not all(np.sum(gear[:, i], axis=0) == 1 for i in range(gear.shape[1])):
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
