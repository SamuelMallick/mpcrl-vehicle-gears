from typing import Literal
import gymnasium as gym
import numpy as np
from vehicle import Vehicle


class VehicleTracking(gym.Env):
    """An environment simulating a vehicle tracking a reference trajectory.
    penalizing tracking errors and fuel consumption.

    Parameters
    ----------
    vehicle : Vehicle
        The vehicle to be controlled.
    trajectory_type : Literal["type_1", "type_2", "type_3"], optional
        The type of reference trajectory to track, by default "type_1".
        Type_1 trajectories are constance velocity and fixed initial
        conditions, while the others are randomized.
        Type_2 trajectories have less aggressive velocity changes and
        less variable initial conditions, with respect to type_3."""

    ts = 1  # sample time (s)
    alpha = 0  # road inclination (rad)

    gamma = 0.1  # weight for tracking in cost
    Q = np.array([[1, 0], [0, 0.1]])  # tracking cost weight

    def __init__(
        self,
        vehicle: Vehicle,
        episode_len: int,
        prediction_horizon: int,
        trajectory_type: Literal["type_1", "type_2", "type_3"] = "type_1",
    ):
        super().__init__()
        self.vehicle = vehicle
        self.x = self.vehicle.x
        self.episode_len = episode_len
        self.prediction_horizon = prediction_horizon
        self.trajectory_type = trajectory_type

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        if options is not None and "x" in options:
            self.vehicle.x = options["x"]
            self.x = self.vehicle.x
        else:
            self.vehicle.x = np.array(
                [[0], [self.np_random.uniform(Vehicle.v_min + 5, Vehicle.v_max - 5)]]
            )
            self.x = self.vehicle.x

        self.x_ref = self.generate_x_ref(trajectory_type=self.trajectory_type)
        self.counter = 0

        return self.x, {}

    def reward(self, x: np.ndarray, fuel: float) -> float:
        return (
            self.gamma
            * (x - self.x_ref[self.counter]).T
            @ self.Q
            @ (x - self.x_ref[self.counter])
            + fuel
        )

    def step(
        self, action: tuple[float, float, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        T_e, F_b, gear = action
        x, fuel, T_e, w_e = self.vehicle.step(T_e, F_b, gear, self.ts, self.alpha)
        r = self.reward(self.x, fuel)
        self.x = x
        self.counter += 1
        return (
            self.x,
            r.item(),
            False,
            False,
            {
                "fuel": fuel,
                "T_e": T_e,
                "w_e": w_e,
                "x_ref": self.x_ref[self.counter - 1],
                "gear": gear,
            },
        )

    def get_x_ref_prediction(self, horizon: int) -> np.ndarray:
        return self.x_ref[self.counter : self.counter + horizon]

    def generate_x_ref(
        self, trajectory_type: Literal["type_1", "type_2", "type_3"] = "type_1"
    ):
        len = self.episode_len + self.prediction_horizon + 1
        if trajectory_type == "type_1":
            d_ref = (self.x[0] + 20 + 20 * self.ts * np.arange(0, len)).reshape(
                len, 1, 1
            )
            v_ref = np.full((len, 1, 1), 20)
            return np.concatenate((d_ref, v_ref), axis=1)
        else:
            x_ref = np.zeros((len, 2, 1))
            if trajectory_type == "type_2":
                v = self.np_random.uniform(15, 25)
                d = 0.0
                slopes = self.np_random.uniform(-0.6, 0.6, size=3)
            else:
                v = self.np_random.uniform(Vehicle.v_min + 5, Vehicle.v_max - 5)
                d = self.np_random.uniform(-50, 50)
                slopes = self.np_random.uniform(2, 2, size=3)

            x_ref[0] = np.array([[d], [v]])
            change_points = np.sort(self.np_random.integers(0, 100, size=5))

            v_clip_range = [5, 35]

            for k in range(change_points[0]):
                x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
            for k in range(change_points[0], change_points[1]):
                v = np.clip(v + slopes[0], v_clip_range[0], v_clip_range[1])
                x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
            for k in range(change_points[1], change_points[2]):
                v = np.clip(v + slopes[1], v_clip_range[0], v_clip_range[1])
                x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
            for k in range(change_points[2], change_points[3]):
                v = np.clip(v + slopes[2], v_clip_range[0], v_clip_range[1])
                x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
            for k in range(change_points[3], len - 1):
                x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])

            return x_ref
