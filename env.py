import gymnasium as gym
import numpy as np
from vehicle import Vehicle


class VehicleTracking(gym.Env):

    ts = 1  # sample time (s)
    alpha = 0  # road inclination (rad)

    gamma = 0.1  # weight for tracking in cost
    Q = np.array([[1, 0], [0, 0.1]])  # tracking cost weight

    def __init__(self, vehicle: Vehicle):
        # TODO add docstring
        super().__init__()
        self.vehicle = vehicle
        self.x = self.vehicle.x

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # TODO add docstring
        super().reset(seed=seed, options=options)
        if options is not None and "x" in options:
            self.vehicle.x = options["x"]
            self.x = self.vehicle.x
        else:
            self.vehicle.x = np.array(
                [[0], [self.np_random.uniform(Vehicle.v_min + 5, Vehicle.v_max - 5)]]
            )
            self.x = self.vehicle.x

        # d_ref = (self.x[0] + 20 + 20 * self.ts * np.arange(0, 500)).reshape(
        #     500, 1, 1
        # )  # TODO get rid of hard code 500
        # v_ref = np.full((500, 1, 1), 20)
        # self.x_ref = np.concatenate((d_ref, v_ref), axis=1)
        self.x_ref = self.generate_random_x_ref()
        self.counter = 0

        return self.x, {}

    def reward(self, x: np.ndarray, fuel: float) -> float:
        # TODO add docstring
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
        # TODO add docstring
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
        # TODO add docstring
        return self.x_ref[self.counter : self.counter + horizon]

    def generate_random_x_ref(self):
        # TODO add docstring
        len = 150  # TODO get rid of hard code 500
        x_ref = np.zeros((len, 2, 1))
        v = self.np_random.uniform(Vehicle.v_min + 5, Vehicle.v_max - 5)
        # v = self.np_random.uniform(15, 25)
        d = self.np_random.uniform(-50, 50)
        # d = 0
        x_ref[0] = np.array([[d], [v]])
        change_points = np.sort(self.np_random.integers(0, 100, size=5))
        # change_points = np.sort(self.np_random.integers(-5, 5, size=4))
        slopes = self.np_random.uniform(2, 2, size=3)
        # slopes = self.np_random.uniform(-0.6, 0.6, size=3)

        for k in range(change_points[0]):  # TODO remove hard coded clipping
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(change_points[0], change_points[1]):
            v = max(min(35, v + slopes[0]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(change_points[1], change_points[2]):
            v = max(min(35, v + slopes[1]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(change_points[2], change_points[3]):
            v = max(min(35, v + slopes[2]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(change_points[3], len - 1):
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])

        return x_ref
