from env import VehicleTracking
import numpy as np
from csnlp.wrappers.mpc.mpc import Mpc

class Agent:

    fuel: list[list[float]] = []
    engine_torque: list[list[float]] = []
    engine_speed: list[list[float]] = []
    x_ref: list[np.ndarray] = []
    x_ref_predicition: np.ndarray = np.empty((0, 2, 1))  # TODO is right shape?

    def __init__(self, mpc: Mpc):
        self.mpc = mpc

    def get_action(self, state: np.ndarray) -> tuple[float, float, int]:
        sol = self.mpc.solve({"x_0": state, "x_ref": self.x_ref_predicition.T.reshape(2, -1)})
        # TODO check success
        T_e = sol.vals['T_e'].full()[0, 0]
        F_b = sol.vals['F_b'].full()[0, 0]
        gear = np.argmax(sol.vals['gear'].full(), 0)[0]
        return T_e, F_b, gear

    def evaluate(self, env: VehicleTracking, episodes, seed=0) -> tuple[np.ndarray, dict]:
        # TODO add docstring

        # self.reset()
        returns = np.zeros(episodes)
        self.on_validation_start()

        for episode in range(episodes):
            state, _ = env.reset(seed=seed)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env)

            while not (truncated or terminated):
                action = self.get_action(state)
                state, reward, truncated, terminated, info = env.step(action)
                self.on_env_step(env, episode, info)

                returns[episode] += reward
                timestep += 1
                # self.on_timestep_end()

            # self.on_episode_end()

        # self.on_validation_end()
        return returns, {
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
        }

    def on_validation_start(self):
        self.fuel = []
        self.engine_torque = []
        self.engine_speed = []
        self.x_ref = []

    def on_episode_start(self, env: VehicleTracking):
        self.fuel.append([])
        self.engine_torque.append([])
        self.engine_speed.append([])
        self.x_ref.append(np.empty((0, 2, 1)))
        self.x_ref_predicition = env.get_x_ref_prediction(self.mpc.prediction_horizon+1)

    def on_env_step(self, env: VehicleTracking, episode: int, info: dict):
        self.fuel[episode].append(info["fuel"])
        self.engine_torque[episode].append(info["T_e"])
        self.engine_speed[episode].append(info["w_e"])
        self.x_ref[episode] = np.concatenate(
            (self.x_ref[episode], info["x_ref"].reshape(1, 2, 1))
        )
        self.x_ref_predicition = env.get_x_ref_prediction(self.mpc.prediction_horizon+1)

