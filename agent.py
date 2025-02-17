from env import VehicleTracking
import numpy as np
from csnlp.wrappers.mpc.mpc import Mpc
from vehicle import Vehicle
from bisect import bisect_right

max_v_per_gear = [
    (Vehicle.w_e_max * Vehicle.r_r * 2 * np.pi) / (Vehicle.z_t[i] * Vehicle.z_f * 60)
    for i in range(6)
]


class Agent:
    fuel: list[list[float]] = []
    engine_torque: list[list[float]] = []
    engine_speed: list[list[float]] = []
    x_ref: list[np.ndarray] = []
    x_ref_predicition: np.ndarray = np.empty((0, 2, 1))
    T_e_prev = Vehicle.T_e_idle
    gear_prev: np.ndarray = np.empty((6, 1))

    def __init__(self, mpc: Mpc):
        self.mpc = mpc

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        raise NotImplementedError

    def evaluate(
        self, env: VehicleTracking, episodes: int, seed: int = 0
    ) -> tuple[np.ndarray, dict]:
        # TODO add docstring

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        # self.reset()
        returns = np.zeros(episodes)
        self.on_validation_start()

        for episode, seed in zip(range(episodes), seeds):
            print(f"Evaluate: Episode {episode}")
            state, _ = env.reset(seed=seed)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(state, env)

            while not (truncated or terminated):
                *action, action_info = self.get_action(state)
                state, reward, truncated, terminated, step_info = env.step(action)
                self.on_env_step(env, episode, timestep, action_info | step_info)

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

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.fuel.append([])
        self.engine_torque.append([])
        self.engine_speed.append([])
        self.x_ref.append(np.empty((0, 2, 1)))
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction(
            self.mpc.prediction_horizon + 1
        )
        self.T_e_prev = Vehicle.T_e_idle
        gear = self.gear_from_velocity(state[1])
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[gear] = 1  # TODO make one line

    def on_env_step(
        self, env: VehicleTracking, episode: int, timestep: int, info: dict
    ):
        self.fuel[episode].append(info["fuel"])
        self.engine_torque[episode].append(info["T_e"])
        self.engine_speed[episode].append(info["w_e"])
        self.x_ref[episode] = np.concatenate(
            (self.x_ref[episode], info["x_ref"].reshape(1, 2, 1))
        )
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction(
            self.mpc.prediction_horizon + 1
        )
        self.T_e_prev = info["T_e"]
        gear = info["gear"]
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[gear] = 1  # TODO make one line

    def gear_from_velocity(self, v: float, current_gear: int | None = None) -> int:
        if current_gear is not None:
            for i in reversed(
                range(max(current_gear - 1, 0), min(current_gear + 2, 6))   # +2 because range is exclusive
            ):  # TODO get rid of loop
                n = Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r
                if v * n * 60 / (2 * np.pi) <= Vehicle.w_e_max + 1e-3:
                    return i
        else:
            for i in reversed(range(6)):
                n = Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r
                if v * n * 60 / (2 * np.pi) <= Vehicle.w_e_max + 1e-3:
                    return i
        raise ValueError("No gear found")


class MINLPAgent(Agent):
    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear_prev": self.gear_prev,
            }
        )
        if not sol.success:
            raise ValueError("MPC failed to solve")
        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        gear = np.argmax(sol.vals["gear"].full(), 0)[0]
        return T_e, F_b, gear, {}


class HeuristicGearAgent(Agent):

    def gear_from_velocity_and_traction(self, v: float, F_trac: float) -> int:
        for i in range(6):
            n = Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r
            if (
                F_trac / n <= Vehicle.T_e_max + 1e-3
                and v * n * 60 / (2 * np.pi) <= Vehicle.w_e_max + 1e-3
            ):
                return i
        raise ValueError("No gear found")

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        # get lowet allowed gear for the given velocity, then use that to limit the traction force
        idx = bisect_right(max_v_per_gear, state[1])
        n = Vehicle.z_f * Vehicle.z_t[idx] / Vehicle.r_r
        F_trac_max = Vehicle.T_e_max * n

        sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "F_trac_max": F_trac_max,
            }
        )

        if not sol.success:
            raise ValueError("MPC failed to solve")
        F_trac = sol.vals["F_trac"].full()[0, 0]
        gear = self.gear_from_velocity_and_traction(state[1], F_trac)
        if F_trac < 0:
            T_e = Vehicle.T_e_idle
            F_b = (
                -F_trac
                + Vehicle.T_e_idle * Vehicle.z_t[gear] * Vehicle.z_f / Vehicle.r_r
            )
        else:
            T_e = (F_trac * Vehicle.r_r) / (Vehicle.z_t[gear] * Vehicle.z_f)
            F_b = 0

        # apply clipping to engine torque
        T_e = (
            np.clip(T_e - self.T_e_prev, -Vehicle.dT_e_max, Vehicle.dT_e_max)
            + self.T_e_prev
        )
        return T_e, F_b, gear, {}
