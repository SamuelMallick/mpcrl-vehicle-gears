import numpy as np

m = 1500  # mass of the vehicle (kg)
C_wind = 0.4071  # wind resistance coefficient
mu = 0.015  # rolling resistance coefficient
g = 9.81  # gravitational acceleration (m/s^2)
r_r = 0.3554  # wheel radius (m)
z_f = 3.39  # final drive ratio
z_t = [4.484, 2.872, 1.842, 1.414, 1.000, 0.742]  # gear ratios

w_e_max = 3000  # maximum engine speed (rpm)
dT_e_max = 100  # maximum engine torque rate (Nm/s)
T_e_idle = 15  # engine idle torque (Nm)
w_e_idle = 900  # engine idle speed (rpm)


class Vehicle:
    def __init__(self, x: np.ndarray = np.array([[0], [20]])):
        """Initialize the vehicle with state containing position and velocity.

        Parameters
        ----------
        x : np.ndarray
            The state of the vehicle, containing position (p) and velocity (v).
            By default p = v = 0.
        """
        self.x = x
        self._T_e: float | None = None  # store previous engine torque

    def step(
        self, T_e: float, F_b: float, gear: int, dt: float, alpha: float = 0
    ) -> tuple[np.ndarray, float, float, float]:
        # TODO add docstring
        if gear < 0 or gear > 5:
            raise ValueError("Gear must be between 0 and 5.")
        if self._T_e is None:
            self._T_e = T_e
        if np.abs(T_e - self._T_e) > dT_e_max:
            print("Engine torque rate exceeded. Clipping.")
            T_e = np.clip(T_e, self._T_e - dT_e_max, self._T_e + dT_e_max)
        if T_e < T_e_idle:
            print("Engine torque below idle. Setting to idle.")
            T_e = T_e_idle
        self._T_e = T_e

        n = z_f * z_t[gear] / r_r
        a = (
            T_e * n / m
            - C_wind * self.x[1] ** 2 / m
            - g * mu * np.cos(alpha)
            - g * np.sin(alpha)
            - F_b / m
        )

        w_e = np.abs(self.x[1]) * n * 60 / (2 * np.pi)
        if w_e > w_e_max:
            raise ValueError("Engine speed exceeds maximum value.")
        if w_e < w_e_idle:
            print("Engine speed below idle. Setting to idle.")
            w_e = w_e_idle

        fuel = self.fuel_rate(T_e, w_e) * dt
        self.x = (
            self.x + np.array([self.x[1], a]) * dt
        )  # TODO make it multiple steps within one for more accuracy
        return self.x, float(fuel), T_e, w_e

    def fuel_rate(self, T_e: float, w_e: float) -> float:
        # TODO add docstring
        return 0.04918 + 0.001897 * w_e + 4.5232e-5 * T_e * w_e
