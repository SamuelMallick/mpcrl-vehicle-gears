from typing import Literal
from csnlp import Solution
import numpy as np
from mpcs.mpc import VehicleMPC
from env import VehicleTracking, PlatoonTracking
from vehicle import Vehicle
import casadi as cs


class Agent:
    """A base class for agents that control vehicle(s).

    Parameters
    ----------
    mpc : VehicleMPC
        The model predictive controller used to solve the optimization problem.
    np_random : np.random.Generator`
        A random number generator for generating random numbers.
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    """

    # the max velocity allowed by each gear while respecting the engine speed limit
    max_v_per_gear = [
        (Vehicle.w_e_max * Vehicle.r_r * 2 * np.pi)
        / (Vehicle.z_t[i] * Vehicle.z_f * 60)
        for i in range(6)
    ]
    min_v_per_gear = [
        (Vehicle.w_e_idle * Vehicle.r_r * 2 * np.pi)
        / (Vehicle.z_t[i] * Vehicle.z_f * 60)
        for i in range(6)
    ]

    n_gears = 6

    def __init__(
        self, mpc: VehicleMPC, np_random: np.random.Generator, multi_starts: int = 1
    ):
        self.mpc = mpc
        self.np_random = np_random
        self.multi_starts = multi_starts
        self.x_ref_predicition: np.ndarray = np.empty((0, 2, 1))

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        """Get the vehicle action, given its current state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle, as a numpy array (position, velocity).

        Returns
        -------
        tuple[float, float, int, dict]
            The action to take, as a tuple of the engine torque, braking force, and gear.
            The action is also accompanied by a dictionary of additional information."""
        raise NotImplementedError

    def clip_action(self, action: tuple[float, float, int]) -> tuple[float, float, int]:
        raise NotImplementedError

    def evaluate(
        self,
        env: VehicleTracking | PlatoonTracking,
        episodes: int,
        seed: int = 0,
        allow_failure: bool = False,
        save_every_episode: bool = False,
        log_progress: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Evaluate the agent on the vehicle tracking environment for a number of episodes.

        Parameters
        ----------
        env : VehicleTracking | PlatoonTracking
            The environment to evaluate the agent on. Can be a single vehicle
            simulation (VehicleTracking) or a platoon simulation (PlatoonTracking).
        episodes : int
            The number of episodes to evaluate the agent for.
        seed : int, optional
            The seed to use for the random number generator, by default 0.
        allow_failure : bool, optional
            If allowed, a failure will cause the episode to be skipped. A
            list of non-failed episodes will be returned in theinfo dict, by default False.
        save_every_episode : bool, optional
            If True, the agent will save the state of the environment at the end
            of each episode, by default False.
        log_progress : bool, optional
            If True, a log file will be created to track the progress of the evaluation,
            By default False.

        Returns
        -------
        tuple[np.ndarray, dict]
            The returns for each episode, and a dictionary of additional information."""

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        if allow_failure:
            valid_episodes = [i for i in range(episodes)]

        # self.reset()
        returns = np.zeros(episodes)
        # self.on_validation_start()

        # create log file
        if log_progress:
            log_file = "log.txt"
            f = open(log_file, "w")
            f.close()

        for episode, seed in zip(range(episodes), seeds):
            print(f"Evaluate: Episode {episode}")
            state, _ = env.reset(seed=seed)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(state, env)

            while not (truncated or terminated):
                print(f"Evaluate: Episode {episode} | Timestep {timestep}")
                if allow_failure:
                    try:
                        *action, action_info = self.get_action(state)
                    except:
                        valid_episodes.remove(episode)
                        break
                else:
                    *action, action_info = self.get_action(state)
                action = self.clip_action(action)
                state, reward, truncated, terminated, step_info = env.step(action)
                self.on_env_step(env, episode, timestep, action_info | step_info)

                returns[episode] += reward

                # log episode progress
                if log_progress:
                    solve_time = self.mpc.solver_time[-1]
                    with open(log_file, "a") as f:
                        f.write(
                            f"Episode {episode} \t | Timestep {timestep} \t | Solver time: {solve_time} \t | Cost: {action_info["cost"] if "cost" in action_info else np.nan}\n"
                        )
                        f.flush()

                timestep += 1
                # self.on_timestep_end()

            # self.on_episode_end(episode, env, save=save_every_episode)

        # self.on_validation_end()
        info = {}
        if allow_failure:
            info["valid_episodes"] = valid_episodes
        return returns, info

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()

    def on_env_step(
        self, env: VehicleTracking, episode: int, timestep: int, info: dict
    ):
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()

    def gear_from_velocity(
        self, v: float, gear_priority: Literal["low", "high", "mid"] = "mid"
    ) -> int:
        """Get a gear a velocity. The gear is chosen such that the engine
        speed limit is not exceeded. The gear is chosen from gears
        that satisfy the engine speed limit.

        Parameters
        ----------
        v : float
            The velocity of the vehicle.
        gear_priority : Literal["low", "high", "mid"], optional
            The priority of the gear to use, by default "mid". If "low", the agent
            will favour lower gears, if "high", the agent will favour higher gears,
            and if "mid", the agent will favour the middle gear that satisfies the
            engine speed limit.

        Returns
        -------
        int
            The gear for the vehicle."""
        if v < Vehicle.v_min or v > Vehicle.v_max:
            if np.isclose(v, Vehicle.v_min) or np.isclose(v, Vehicle.v_max):
                v = np.clip(v, Vehicle.v_min, Vehicle.v_max)
            else:
                raise ValueError("Velocity out of bounds")
        # TODO look at the buffer zone
        valid_gears = [
            (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            <= Vehicle.w_e_max + 1e-6
            and (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            >= Vehicle.w_e_idle - 1e-6
            for i in range(6)
        ]
        valid_indices = [i for i, valid in enumerate(valid_gears) if valid]
        if not valid_indices:
            raise ValueError("No gear found")
        if gear_priority == "high":
            return valid_indices[0]
        if gear_priority == "low":
            return valid_indices[-1]
        return valid_indices[len(valid_indices) // 2]

    def initial_guesses_vals(
        self, state: np.ndarray, num_guesses: int = 5
    ) -> list[dict]:
        """Get initial guesses for optimization variables.
        The initial guesses are generated by sampling from uniform distributions
        within their respective bounds. Additionally, the current state is included
        in the guess of the state trajectory.

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle, as a numpy array (position, velocity).
        num_guesses : int, optional
            The number of initial guesses to generate, by default 5.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing the initial guesses for the optimization variables.
            Each dictionary has keys "x", "T_e", "F_b", "w_e", and "gear" corresponding to the state,
            engine torque, braking force, engine speed, and gear respectively."""
        starts = []
        for _ in range(num_guesses):
            d = {}
            d["x"] = np.concatenate(
                (
                    state,
                    np.vstack(
                        (
                            self.np_random.uniform(
                                state[0] - Vehicle.v_min * self.mpc.prediction_horizon,
                                state[0] + Vehicle.v_max * self.mpc.prediction_horizon,
                                (1, self.mpc.prediction_horizon),
                            ),  # TODO bring in ts
                            self.np_random.uniform(
                                self.min_v_per_gear[0],
                                self.max_v_per_gear[-1],
                                (1, self.mpc.prediction_horizon),
                            ),
                        )
                    ),
                ),
                axis=1,
            )
            d["T_e"] = self.np_random.uniform(
                Vehicle.T_e_idle, Vehicle.T_e_idle, (1, self.mpc.prediction_horizon)
            )
            d["F_b"] = self.np_random.uniform(
                0, Vehicle.F_b_max, (1, self.mpc.prediction_horizon)
            )
            d["w_e"] = self.np_random.uniform(
                Vehicle.w_e_idle, Vehicle.w_e_max, (1, self.mpc.prediction_horizon)
            )
            gears = self.np_random.integers(0, 6, (1, self.mpc.prediction_horizon))
            d["gear"] = np.zeros((6, self.mpc.prediction_horizon))
            for j in range(self.mpc.prediction_horizon):
                d["gear"][gears[0, j], j] = 1
            starts.append(d)
        return starts

    def shift_sol(self, sol) -> Solution:
        for key in sol.vals.keys():
            sol.vals[key] = cs.horzcat(sol.vals[key][:, 1:], sol.vals[key][:, -1:])
        return sol


class SingleVehicleAgent(Agent):

    T_e_prev = Vehicle.T_e_idle
    gear_prev = np.zeros((6, 1))
    prev_sol = None

    def on_episode_start(self, state, env):
        super().on_episode_start(state, env)
        self.T_e_prev = Vehicle.T_e_idle
        self.gear = self.gear_from_velocity(state[1].item())
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[self.gear] = 1

    def on_env_step(self, env, episode, timestep, info):
        super().on_env_step(env, episode, timestep, info)
        self.T_e_prev = info["T_e"]
        self.gear = info["gear"]
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[self.gear] = 1

    def clip_action(self, action):
        if np.abs(np.argmax(self.gear_prev) - action[2]) > 1:
            print("Warning: clipping gear in evaluate")
            action[2] = np.clip(
                action[2],
                np.argmax(self.gear_prev) - 1,
                np.argmax(self.gear_prev) + 1,
            )
        return action


class PlatoonAgent(Agent):

    d = 25  # inter-vehicle distance (m)
    d_arr = np.array([[d], [0]])

    def __init__(
        self,
        mpc: VehicleMPC,
        num_vehicles: int,
        np_random: np.random.Generator,
        multi_starts=1,
    ):
        super().__init__(mpc, np_random, multi_starts)
        self.num_vehicles = num_vehicles
        self.T_e_prev = [Vehicle.T_e_idle for _ in range(num_vehicles)]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(num_vehicles)]
        self.prev_sols = [None for _ in range(num_vehicles)]

    def on_episode_start(self, state, env):
        Agent.on_episode_start(self, state, env)
        self.T_e_prev = [Vehicle.T_e_idle for _ in range(self.num_vehicles)]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(self.num_vehicles)]
        xs = np.split(state, self.num_vehicles, axis=1)
        for i, x in enumerate(xs):
            gear = self.gear_from_velocity(x[1].item())
            self.gear_prev[i][gear] = 1

    def on_env_step(self, env, episode, timestep, info):
        Agent.on_env_step(self, env, episode, timestep, info)
        self.T_e_prev = info["T_e"]
        gears = info["gear"]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(self.num_vehicles)]
        for i, gear in enumerate(gears):
            self.gear_prev[i][gear] = 1

    def get_pars(self, x: np.ndarray, i: int) -> dict:
        pars = {"x_0": x, "T_e_prev": self.T_e_prev[i], "gear_prev": self.gear_prev[i]}
        if i == 0:
            pars["x_ref"] = self.x_ref_predicition.T.reshape(2, -1)
        else:
            pars["x_ref"] = self.prev_sols[i - 1].vals["x"] - self.d_arr
        if i < self.num_vehicles - 1:
            if (
                self.prev_sols[i + 1] is not None
            ):  # for first solution behind constraint not considered
                pars["p_b"] = self.prev_sols[i + 1].vals["x"][0, :]
        if i > 0:
            pars["p_a"] = self.prev_sols[i - 1].vals["x"][0, :]
        return pars

    def clip_action(self, action):
        for i in range(self.num_vehicles):
            if np.abs(np.argmax(self.gear_prev[i]) - action[2][i]) > 1:
                print("Warning: clipping gear in evaluate")
                action[2][i] = np.clip(
                    action[2][i],
                    np.argmax(self.gear_prev[i]) - 1,
                    np.argmax(self.gear_prev[i]) + 1,
                )
        return action
