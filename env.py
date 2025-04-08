from typing import Literal
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
from vehicle import Vehicle

DEBUG = False
PLOT = False


class VehicleTracking(gym.Env):
    """An environment simulating a vehicle tracking a reference trajectory.
    penalizing tracking errors and fuel consumption.

    Parameters
    ----------
    vehicle : Vehicle
        The vehicle to be controlled.
    prediction_horizon : int
        The length of the prediction horizon used by the MPC that
        interacts with the env.
    windy : bool, optional
        Whether to include wind in the simulation, by default False.
    trajectory_type : Literal["type_1", "type_2", "type_3"], optional
        The type of reference trajectory to track, by default "type_1".
        Type_1 trajectories are constance velocity and fixed initial
        conditions, while the others are randomized.
        Type_2 trajectories have less aggressive velocity changes and
        less variable initial conditions, with respect to type_3.
    terminate_on_distance : bool, optional
        Whether to terminate the episode if the vehicle is too far from
        the reference trajectory, by default False. The distance is
        100m."""

    ts = 1  # sample time (s)
    alpha = 0  # road inclination (rad)

    gamma = 0.01  # weight for tracking in cost
    Q = np.array([[1, 0], [0, 0.1]])  # tracking cost weight

    wind_speed_range = (8, 14)  # wind speed range (m/s)
    wind_delta = 0.1  # wind speed change rate (m/s)

    wind_list = []

    def __init__(
        self,
        vehicle: Vehicle,
        prediction_horizon: int,
        windy: bool = False,
        trajectory_type: Literal["type_1", "type_2", "type_3"] = "type_1",
        terminate_on_distance: bool = False,
    ):
        super().__init__()
        self.vehicle = vehicle
        self.x = self.vehicle.x
        self.prediction_horizon = prediction_horizon
        self.trajectory_type = trajectory_type
        self.windy = windy
        self.terminate_on_distance = terminate_on_distance

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

        # generate reference trajectory
        self.x_ref = self.init_x_ref(self.x)

        if self.windy:
            self.wind = self.np_random.uniform(
                self.wind_speed_range[0], self.wind_speed_range[1], size=1
            )

        self.counter = 0

        return self.x, {}

    def reward(self, x: np.ndarray, x_ref: np.ndarray, fuel: float) -> float:
        return (  # first element in x_ref is the current desired state
            self.gamma * (x - x_ref).T @ self.Q @ (x - x_ref) + fuel
        )

    def step(
        self, action: tuple[float, float, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        if self.terminate_on_distance and np.abs(self.x[0] - self.x_ref[0, 0]) > 100:
            terminated = True
        T_e, F_b, gear = action
        prev_x = self.x
        prev_x_ref = self.x_ref[0]
        x, fuel, T_e, w_e = self.vehicle.step(
            T_e,
            F_b,
            gear,
            self.ts,
            self.alpha,
            wind_speed=self.wind.item() if self.windy else 0,
        )
        r = self.reward(prev_x, prev_x_ref, fuel)
        self.x = x
        self.counter += 1
        self.next_x_ref(self.trajectory_type)
        if self.windy:
            self.next_wind()
        return (
            self.x,
            r.item(),
            False,
            terminated,
            {
                "fuel": fuel,
                "T_e": T_e,
                "w_e": w_e,
                "x_ref": prev_x_ref,
                "gear": gear,
                "x": prev_x,
                "wind": self.wind if self.windy else 0,
            },
        )

    def get_x_ref_prediction(self) -> np.ndarray:
        return self.x_ref

    def init_x_ref(
        self, x: np.ndarray, v_clip_range: list[float] = [5, 28]
    ) -> np.ndarray:
        """Initializes the reference trajectory of length prediction_horizon + 1.

        Parameters
        ----------
        x : np.ndarray
            Initial state of the trajectory.
        v_clip_range : list[float], optional
            Range of velocity clipping, by default [5, 28].

        Returns
        -------
        np.ndarray
            The reference trajectory.
        """
        x_ref = np.empty((self.prediction_horizon + 1, 2, 1))
        self.a = self.np_random.uniform(-0.6, 0.6)
        x_ref[0] = x
        for i in range(1, self.prediction_horizon + 1):
            x_ref[i, 0] = x_ref[i - 1, 0] + self.ts * x_ref[i - 1, 1]
            v = x_ref[i - 1, 1] + self.ts * self.a
            v = np.clip(v, v_clip_range[0], v_clip_range[1])
            x_ref[i, 1] = v
        return x_ref

    def next_x_ref(
        self,
        trajectory_type: Literal["type_1", "type_2", "type_3"] = "type_1",
        accel_change_freq: float = 1.0 / 20,
        v_clip_range: list[float] = [5, 28],
    ) -> None:
        """Updates the reference trajectory by removing the first element and
        appending a new one based.

        Parameters
        ----------
        trajectory_type : Literal["type_1", "type_2", "type_3"], optional"
            "type_1" : constant velocity"
            "type_2" : less aggressive velocity changes""
            "type_3" : more aggressive velocity changes" "

        accel_change_freq : float, optional
            Frequency of changing acceleration, by default 1.0 / 20.

        v_clip_range : list[float], optional
            Range of velocity clipping, by default [5, 28].
        """
        if trajectory_type == "type_1":
            d = self.x_ref[-1, 0, 0]
            v = self.x_ref[-1, 1, 0]
            self.x_ref = np.concatenate(
                (
                    self.x_ref[1:],
                    np.array([d + self.ts * v, v]).reshape(1, 2, 1),
                )
            )
        elif trajectory_type == "type_2":
            if self.np_random.uniform() < accel_change_freq:
                self.a = self.np_random.uniform(-0.6, 0.6)
            d = self.x_ref[-1, 0, 0]
            v = self.x_ref[-1, 1, 0]
            d_ = d + self.ts * v
            v_ = v + self.ts * self.a
            v_ = np.clip(v_, v_clip_range[0], v_clip_range[1])
            self.x_ref = np.concatenate(
                (self.x_ref[1:], np.array([d_, v_]).reshape(1, 2, 1))
            )
        elif trajectory_type == "type_3":
            if self.np_random.uniform() < accel_change_freq:
                self.a = self.np_random.uniform(-3, 3)
            d = self.x_ref[-1, 0, 0]
            v = self.x_ref[-1, 1, 0]
            d_ = d + self.ts * v
            v_ = v + self.ts * self.a
            v_ = np.clip(v_, v_clip_range[0], v_clip_range[1])
            self.x_ref = np.concatenate(
                (self.x_ref[1:], np.array([d_, v_]).reshape(1, 2, 1))
            )

    def next_wind(self) -> None:
        self.wind += self.ts * self.np_random.uniform(-self.wind_delta, self.wind_delta)
        self.wind = np.clip(
            self.wind, self.wind_speed_range[0], self.wind_speed_range[1]
        )
        self.wind_list.append(self.wind.item())


class PlatoonTracking(VehicleTracking):

    # TODO doc string
    def __init__(
        self,
        vehicles: list[Vehicle],
        prediction_horizon: int,
        inter_vehicle_distance: float,
        windy: bool = False,
        trajectory_type: Literal["type_1", "type_2", "type_3"] = "type_1",
        infinite_episodes: bool = False,
    ):
        gym.Env.__init__(self)
        self.d = inter_vehicle_distance
        self.d_arr = np.array([[self.d], [0]])
        self.vehicles = vehicles
        self.x = np.concatenate([v.x for v in self.vehicles], axis=1)
        self.prediction_horizon = prediction_horizon
        self.trajectory_type = trajectory_type
        self.windy = windy
        self.infinite_episodes = infinite_episodes

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        gym.Env.reset(self, seed=seed, options=options)
        if options is not None and "x" in options:
            self.x = options["x"]
            xs = np.split(self.x, len(self.vehicles), axis=1)
            for i, v in enumerate(self.vehicles):
                v.x = xs[i]
        else:
            vel = self.np_random.uniform(Vehicle.v_min + 5, Vehicle.v_max - 5)
            for i, v in enumerate(self.vehicles):
                v.x = np.array([[-self.d * i], [vel]])
            self.x = np.concatenate([v.x for v in self.vehicles], axis=1)

        # generate reference trajectory - starting from first vehicle state
        self.x_ref = self.init_x_ref(self.vehicles[0].x)

        if self.windy:
            raise NotImplementedError
            self.wind = self.np_random.uniform(
                self.wind_speed_range[0], self.wind_speed_range[1], size=1
            )

        self.counter = 0

        return self.x, {}

    def reward(
        self, x: list[np.ndarray], x_ref: np.ndarray, fuel: list[float]
    ) -> float:
        cost = self.gamma * (x[0] - x_ref).T @ self.Q @ (x[0] - x_ref)
        cost += sum(
            [
                self.gamma
                * (x[i] - x[i - 1] + self.d_arr).T
                @ self.Q
                @ (x[i] - x[i - 1] + self.d_arr)
                for i in range(1, len(self.vehicles))
            ]
        )
        cost += sum(fuel)
        return cost

    def step(
        self, action: tuple[np.ndarray, np.ndarray, list[int]]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        prev_x = np.split(self.x, len(self.vehicles), axis=1)
        prev_x_ref = self.x_ref[0]
        terminated = False
        # TODO add in desired distance here
        if (
            self.infinite_episodes
            and np.max(
                [np.abs(prev_x[0][0] - prev_x_ref[0, 0])]
                + [
                    np.abs(prev_x[i][0] - prev_x[i - 1][0] + self.d)
                    for i in range(1, len(self.vehicles))
                ]
            )
            > 100
        ):
            terminated = True
        T_e, F_b, gear = action
        T_e_ = np.split(T_e, len(self.vehicles))
        F_b_ = np.split(F_b, len(self.vehicles))
        fuel = [0] * len(self.vehicles)
        x = np.empty_like(self.x)
        T_e = np.empty_like(T_e)
        w_e = np.empty_like(T_e)
        for i, v in enumerate(self.vehicles):
            x[:, [i]], fuel[i], T_e[i], w_e[i] = v.step(
                T_e_[i].item(),
                F_b_[i].item(),
                gear[i],
                self.ts,
                self.alpha,
                # wind_speed=self.wind.item() if self.windy else 0,
            )
        r = self.reward(prev_x, prev_x_ref, fuel)
        self.x = x
        self.counter += 1
        self.next_x_ref(self.trajectory_type)
        # if self.windy:
        #     self.next_wind()
        return (
            self.x,
            r.item(),
            False,
            terminated,
            {
                "fuel": fuel,
                "T_e": T_e,
                "w_e": w_e,
                "x_ref": prev_x_ref,
                "gear": gear,
                "x": prev_x,
                # "wind": self.wind if self.windy else 0,
            },
        )
