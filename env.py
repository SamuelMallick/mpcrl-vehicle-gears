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
            self.vehicle.x = np.array([[0], [10]])
            self.x = self.vehicle.x

        d_ref = (self.x[0] + 20 + 20 * self.ts * np.arange(0, 500)).reshape(
            500, 1, 1
        )  # TODO get rid of hard code 500
        v_ref = np.full((500, 1, 1), 20)
        self.x_ref = np.concatenate((d_ref, v_ref), axis=1)
        # self.x_ref = self.generate_random_x_ref()
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
            },
        )

    def get_x_ref_prediction(self, horizon: int) -> np.ndarray:
        # TODO add docstring
        return self.x_ref[self.counter : self.counter + horizon]

    def generate_random_x_ref(self):
        # TODO add docstring
        len = 110  # TODO get rid of hard code 500
        x_ref = np.zeros((len, 2, 1))
        v = 20 + self.np_random.uniform(-5, 5)
        x_ref[0] = np.array([[self.x[0, 0]], [v]])
        change_points = np.sort(self.np_random.integers(-5, 5, size=4))
        slopes = self.np_random.uniform(-0.6, 0.6, size=3)

        for k in range(20 + change_points[0]):
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(20 + change_points[0], 35 + change_points[1]):
            v = max(min(35, v + slopes[0]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(35 + change_points[1], 50 + change_points[2]):
            v = max(min(35, v + slopes[1]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(50 + change_points[2], 70 + change_points[3]):
            v = max(min(35, v + slopes[2]), 5)
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])
        for k in range(70 + change_points[3], len - 1):
            x_ref[k + 1] = np.array([[x_ref[k, 0, 0] + self.ts * v], [v]])

        return x_ref
