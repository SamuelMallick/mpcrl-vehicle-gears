import casadi as cs
from csnlp.wrappers.mpc.mpc import Mpc
from csnlp import Nlp
from vehicle import Vehicle
import numpy as np

m = 1500  # mass of the vehicle (kg)
C_wind = 0.4071  # wind resistance coefficient
mu = 0.015  # rolling resistance coefficient
g = 9.81  # gravitational acceleration (m/s^2)
r_r = 0.3554  # wheel radius (m)
z_f = 3.39  # final drive ratio
z_t = [4.484, 2.872, 1.842, 1.414, 1.000, 0.742]  # gear ratios

w_e_max = 3000  # maximum engine speed (rpm)
T_e_max = 300   # maximum engine torque (Nm)
dT_e_max = 100  # maximum engine torque rate (Nm/s)
T_e_idle = 15   # engine idle torque (Nm)
w_e_idle = 900  # engine idle speed (rpm)

Q = cs.diag([1, 0.1])


def non_linear_model(x, u, dt, alpha):
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


class HybridTrackingMpc(Mpc):
    def __init__(self, prediction_horizon: int):
        nlp = Nlp[cs.SX](sym_type="SX")
        super().__init__(nlp, prediction_horizon)

        x, _ = self.state("x", 2)

        T_e, _ = self.action("T_e", 1, lb=T_e_idle, ub=T_e_max)
        self.constraint("engine_torque_rate_ub", T_e[:-1] - T_e[1:], "<=", dT_e_max)
        self.constraint("engine_torque_rate_lb", T_e[:-1] - T_e[1:], ">=", -dT_e_max)

        F_b, _ = self.action("F_b", 1, lb=0)

        gear, _ = self.action("gear", 6, discrete=True, lb=0, ub=1)
        self.constraint("gear_constraint", cs.sum1(gear), "==", 1)

        w_e, _, _ = self.variable("w_e", (1, prediction_horizon), lb=w_e_idle, ub=w_e_max)
        n = (z_f / r_r) * sum([z_t[i] * gear[i, :] for i in range(6)])
        self.constraint("engine_speed", w_e, "==", x[1, :-1] * n * 60 / (2 * np.pi))

        

        x_ref = self.parameter("x_ref", (2, prediction_horizon+1))
        self.set_nonlinear_dynamics(lambda x, u: non_linear_model(x, u, 0.1, 0))
        self.minimize(
            sum(
                [
                    cs.mtimes([(x[:, i] - x_ref[:, i]).T, Q, x[:, i] - x_ref[:, i]])
                    for i in range(prediction_horizon + 1)
                ]
            )
        )
        self.init_solver({}, solver="bonmin")
