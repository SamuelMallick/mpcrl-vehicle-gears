import numpy as np


class Vehicle:
    """A class for a single vehicle with longitudinal motion. Both the vehicle and
    engine models are based on the paper Vehicle Speed and Gear Position Co-Optimization
    for Energy-Efficient Connected and Autonomous Vehicles by Y. Shao et al. (2020).

    Parameters
    ----------
    x : np.ndarray
        The initial state of the vehicle, containing position (p) and velocity (v).
    raise_errors : bool
        Whether to raise errors if engine speed, requested engine torque, or velocity
        exceeds limits and rates. If false, the values are clipped to respect limits
        and rates."""

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
    T_e_max = 300  # maximum engine torque (Nm)

    # maximum and minimu velocity: calculated from maximum engine speed: w_e = (z_f * z_t[gear] * 60)/(r_r * 2 * pi)
    v_max = (w_e_max * 2 * np.pi * r_r) / (60 * z_f * z_t[-1])
    v_min = (w_e_idle * 2 * np.pi * r_r) / (60 * z_f * z_t[0])

    _T_e: float | None = None  # store previous engine torque

    def __init__(
        self, x: np.ndarray = np.array([[0], [20]]), raise_errors: bool = False
    ):
        if x.shape != (2, 1):
            raise ValueError("Initial vehicle state must be of size (2,1).")
        self.x = x
        self.raise_errors = raise_errors

    def step(
        self,
        T_e: float,
        F_b: float,
        gear: int,
        dt: float,
        alpha: float = 0,
        substeps: int = 1,
    ) -> tuple[np.ndarray, float, float, float]:
        """Simulate the vehicle motion for a timestep of length dt.

        Parameters
        ----------
        T_e : float
            Engine torque (Nm).
        F_b : float
            Braking force (N).
        gear : int
            Gear position (0-5).
        dt : float
            Timestep length (s).
        alpha : float, optional
            Road gradient (radians), by default 0.
        substeps : int, optional
            Number of substeps to simulate within one timestep, by default 1.

        Returns
        -------
        tuple[np.ndarray, float, float, float]
            The new state of the vehicle, fuel consumption over timestep, actual engine torque, and actual engine speed.
        """
        if gear < 0 or gear > 5:
            raise ValueError("Gear must be between 0 and 5.")
        if self._T_e is None:
            self._T_e = T_e
        if np.abs(T_e - self._T_e) > self.dT_e_max:
            if self.raise_errors:
                raise ValueError("Engine torque rate exceeded.")
            T_e = np.clip(T_e, self._T_e - self.dT_e_max, self._T_e + self.dT_e_max)
        if T_e < self.T_e_idle or T_e > self.T_e_max:
            if self.raise_errors:
                raise ValueError("Engine torque exceeds limits.")
            T_e = np.clip(T_e, self.T_e_idle, self.T_e_max)
        self._T_e = T_e

        n = self.z_f * self.z_t[gear] / self.r_r
        a = (
            T_e * n / self.m
            - self.C_wind * self.x[1, 0] ** 2 / self.m
            - self.g * self.mu * np.cos(alpha)
            - self.g * np.sin(alpha)
            - F_b / self.m
        )

        w_e = np.abs(self.x[1, 0]) * n * 60 / (2 * np.pi)
        if w_e > self.w_e_max + 1:  # +1 for numerical tolerance
            raise ValueError(
                "Engine speed exceeds maximum value."
            )  # always raising error on w_e, as signifies an invalid gear chosen
        if w_e < self.w_e_idle:
            if self.raise_errors:
                raise ValueError("Engine speed below idle.")
            w_e = self.w_e_idle

        fuel = self.fuel_rate(T_e, w_e) * dt

        if self.x[1] + dt * a < self.v_min or self.x[1] + dt * a > self.v_max:
            if self.raise_errors:
                raise ValueError("Velocity exceeds limits.")
            if self.x[1] + dt * a < self.v_min:
                a = (self.v_min - self.x[1, 0]) / dt
            else:
                a = (self.v_max - self.x[1, 0]) / dt
            T_e = (1 / n) * (
                self.g * self.mu * np.cos(alpha)
                + self.g * np.sin(alpha)
                + F_b
                + self.C_wind * self.x[1, 0] ** 2
                + self.m * a
            )
            if T_e < self.T_e_idle or T_e > self.T_e_max:
                raise ValueError(
                    "Engine torque exceeds limits after adjusting for speed."
                )

        for _ in range(substeps):
            self.x = self.x + np.array([[self.x[1, 0]], [a]]) * dt / substeps
        return self.x, float(fuel), T_e, w_e

    def fuel_rate(self, T_e: float, w_e: float) -> float:
        """Calculate the fuel consumption rate of the engine.

        Parameters
        ----------
        T_e : float
            Engine torque (Nm).
        w_e : float
            Engine speed (rpm).

        Returns
        -------
        float
            Fuel consumption rate (g/s)."""
        return (
            0.04918 + 0.001897 * w_e + 4.5232e-5 * T_e * w_e
        )  # constants taken from the paper cited above
