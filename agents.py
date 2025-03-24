from typing import Literal

from csnlp import Solution
import torch
from config_files.base import ConfigDefault
from env import VehicleTracking
import numpy as np
from csnlp.wrappers.mpc.mpc import Mpc
from mpc import HybridTrackingMpc, TrackingMpc
from network import DRQN, ReplayMemory, Transition
from utils.running_mean_std import RunningMeanStd
from vehicle import Vehicle
from bisect import bisect_right
import pickle
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# the max velocity allowed by each gear while respecting the engine speed limit
max_v_per_gear = [
    (Vehicle.w_e_max * Vehicle.r_r * 2 * np.pi) / (Vehicle.z_t[i] * Vehicle.z_f * 60)
    for i in range(6)
]
min_v_per_gear = [
    (Vehicle.w_e_idle * Vehicle.r_r * 2 * np.pi) / (Vehicle.z_t[i] * Vehicle.z_f * 60)
    for i in range(6)
]


class Agent:
    """A base class for agents that control the vehicle. interact with the
    environment.

    Parameters
    ----------
    mpc : Mpc
        The model predictive controller used to solve the optimization problem.
        Can be a mixed-integer optimization problem, solving for gears as well
        as continuous variables, or a parametric optimization problem with the
        gears as parameters. The specific MPC used depends on the subclass."""

    # data tracked over episodes of training or evaluation
    fuel: list[list] = []
    engine_torque: list[list] = []
    engine_speed: list[list] = []
    x_ref: list[np.ndarray] = []
    x_ref_predicition: np.ndarray = np.empty((0, 2, 1))
    T_e_prev = Vehicle.T_e_idle
    gear_prev: np.ndarray = np.empty((6, 1))

    def __init__(self, mpc: Mpc):
        self.mpc = mpc

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

    def evaluate(
        self,
        env: VehicleTracking,
        episodes: int,
        seed: int = 0,
        allow_failure: bool = False,
        save_every_episode: bool = False,
        log_progress: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Evaluate the agent on the vehicle tracking environment for a number of episodes.

        Parameters
        ----------
        env : VehicleTracking
            The vehicle tracking environment to evaluate the agent on.
        episodes : int
            The number of episodes to evaluate the agent for.
        seed : int, optional
            The seed to use for the random number generator, by default 0.
        allow_failure : bool, optional
            If allowed, a failure will cause the episode to be skipped. A
            list of non-failed episodes will be returned in the
            info dict, by default False.
        save_every_episode : bool, optional
            If True, the agent will save the state of the environment at the end
            of each episode, by default False.
        log_progress : bool, optional
            If True, log the episode number to keep track of the progress, by default
            False.

        Returns
        -------
        tuple[np.ndarray, dict]
            The returns for each episode, and a dictionary of additional information."""

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        if allow_failure:
            valid_episodes = [i for i in range(episodes)]

        # self.reset()
        returns = np.zeros(episodes)
        self.on_validation_start()

        # create log file
        if log_progress:
            log_file = "log.txt"
            f = open(log_file, "w")
            f.write(f"Eval begun at {str(datetime.now())}\n")
            f.close()

        for episode, seed in zip(range(episodes), seeds):
            print(f"Evaluate: Episode {episode}")
            state, _ = env.reset(seed=seed)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(state, env)

            while not (truncated or terminated):
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
                    if action_info["solver"] == "primary":
                        solve_time = self.mpc.solver_time[-1]
                    elif action_info["solver"] == "backup":
                        solve_time = self.backup_mpc.solver_time[-1]
                    with open(log_file, "a") as f:
                        f.write(
                            f"Episode {episode} \t | Timestep {timestep} \t | Solver: {action_info['solver']} \t | Solver time: {solve_time}\n"
                        )
                        f.flush()

                timestep += 1
                # self.on_timestep_end()

            self.on_episode_end(episode, env, save=save_every_episode)

        # self.on_validation_end()
        info = {
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
        }
        if allow_failure:
            info["valid_episodes"] = valid_episodes
        return returns, info

    def clip_action(self, action):
        if np.abs(np.argmax(self.gear_prev) - action[2]) > 1:
            action[2] = np.clip(
                action[2],
                np.argmax(self.gear_prev) - 1,
                np.argmax(self.gear_prev) + 1,
            )
        return action

    def on_episode_end(self, episode: int, env: VehicleTracking, save: bool = False):
        if save:
            with open(f"episode_{episode}.pkl", "wb") as f:
                pickle.dump(
                    {
                        "fuel": self.fuel,
                        "T_e": self.engine_torque,
                        "w_e": self.engine_speed,
                        "x_ref": self.x_ref,
                        "X": env.observations,
                        "U": env.actions,
                        "R": env.rewards,
                        "mpc_solve_time": self.mpc.solver_time,
                    },
                    f,
                )

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
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()
        self.T_e_prev = Vehicle.T_e_idle
        gear = self.gear_from_velocity(state[1].item())
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[gear] = 1

    def on_env_step(
        self, env: VehicleTracking, episode: int, timestep: int, info: dict
    ):
        self.fuel[episode].append(info["fuel"])
        self.engine_torque[episode].append(info["T_e"])
        self.engine_speed[episode].append(info["w_e"])
        self.x_ref[episode] = np.concatenate(
            (self.x_ref[episode], info["x_ref"].reshape(1, 2, 1))
        )
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()
        self.T_e_prev = info["T_e"]
        gear = info["gear"]
        self.gear_prev = np.zeros((6, 1))
        self.gear_prev[gear] = 1

    def gear_from_velocity(self, v: float) -> int:
        """Get the gear that the vehicle should be in, given its velocity. The gear
        is chosen such that the engine speed limit is not exceeded. The gear is
        chosen as the middle gear that satisfies the engine speed limit.

        Parameters
        ----------
        v : float
            The velocity of the vehicle.

        Returns
        -------
        int
            The gear that the vehicle should be in."""
        if v < Vehicle.v_min or v > Vehicle.v_max:
            if np.isclose(v, Vehicle.v_min) or np.isclose(v, Vehicle.v_max):
                v = np.clip(v, Vehicle.v_min, Vehicle.v_max)
            else:
                raise ValueError("Velocity out of bounds")
        valid_gears = [
            (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            <= Vehicle.w_e_max
            and (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            >= Vehicle.w_e_idle
            for i in range(6)
        ]
        valid_indices = [i for i, valid in enumerate(valid_gears) if valid]
        if not valid_indices:
            raise ValueError("No gear found")
        return valid_indices[len(valid_indices) // 2]


class MINLPAgent(Agent):
    """An agent that uses a mixed-integer nonlinear programming solver to solve
    the optimization problem. The solver is used to find the optimal engine torque,
    braking force, and gear for the vehicle.

    Parameters
    ----------
    mpc : HybridTrackingMpc
        The model predictive controller used to solve the optimization problem.
    backup_mpc : HybridTrackingMpc, optional
        A backup model predictive controller used when the primary MPC fails to
        solve the optimization problem, by default None."""

    def __init__(
        self, mpc: HybridTrackingMpc, backup_mpc: HybridTrackingMpc | None = None
    ):
        super().__init__(mpc)
        self.backup_mpc = backup_mpc

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        solver = "primary"
        sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear_prev": self.gear_prev,
            }
        )
        if not sol.success:
            solver = "backup"
            sol = self.backup_mpc.solve(
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
        return T_e, F_b, gear, {"solver": solver}


class HeuristicGearAgent(Agent):
    """An agent that solves a continuous optimization problem with an approximate
    model, considering only continuous variables. Then it uses a heuristic to
    determine the gear to use, based on the engine speed and traction force.

    Parameters
    ----------
    mpc : TrackingMpc
        The model predictive controller used to solve the optimization problem.
    gear_priority : Literal["low", "high", "mid"], optional
        The priority of the gear to use, by default "mid". If "low", the agent
        will favour lower gears, if "high", the agent will favour higher gears,
        and if "mid", the agent will favour the middle gear that satisfies the
        engine speed limit."""

    def __init__(
        self, mpc: TrackingMpc, gear_priority: Literal["low", "high", "mid"] = "mid"
    ):
        self.gear_priority = gear_priority
        super().__init__(mpc)

    def gear_from_velocity_and_traction(self, v: float, F_trac: float) -> int:
        # TODO enforce gear switch of only 1
        valid_gears = [
            (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            <= Vehicle.w_e_max + 1e-3
            and (v * Vehicle.z_f * Vehicle.z_t[i] * 60) / (2 * np.pi * Vehicle.r_r)
            >= Vehicle.w_e_idle - 1e-3
            and F_trac / (Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r)
            <= Vehicle.T_e_max + 1e-3
            and (
                (
                    F_trac
                    < Vehicle.T_e_idle * (Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r)
                )
                or F_trac / (Vehicle.z_f * Vehicle.z_t[i] / Vehicle.r_r)
                >= Vehicle.T_e_idle - 1e-3
            )
            for i in range(6)
        ]
        valid_indices = [i for i, valid in enumerate(valid_gears) if valid]
        if not valid_indices:
            raise ValueError("No gear found")
        if self.gear_priority == "high":
            return valid_indices[0]
        if self.gear_priority == "low":
            return valid_indices[-1]
        return valid_indices[len(valid_indices) // 2]

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        # get highest allowed gear for the given velocity, then use that to limit the traction force
        idx = bisect_right(max_v_per_gear, state[1].item())
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


class LearningAgent(Agent):
    """An agent that uses a learning-based policy to select the gear-shift
    schedule. An NLP-based MPC controller then solves for the continuous
    variables, given the gear-shift schedule."""

    infeasible: list[list[float]] = (
        []
    )  # flag to store if the gear choice was infeasible

    def __init__(
        self,
        mpc: HybridTrackingMpc,
        np_random: np.random.Generator,
        config: ConfigDefault,
    ):
        super().__init__(mpc)
        self.first_timestep: bool = (
            True  # gets reset gets set to True in on_episode_start
        )
        self.np_random = np_random
        self.n_gears = 6
        self.steps_done = 0

        self.expert_mpc = config.expert_mpc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("Using GPU")

        # hyperparameters
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        # archticeture
        self.clipping = config.clip
        self.n_states = config.n_states
        self.n_hidden = config.n_hidden
        self.n_actions = config.n_actions
        self.n_layers = config.n_layers
        self.N = config.N
        self.normalize = config.normalize

        # seeded initialization of networks
        seed = np_random.integers(0, 2**32 - 1)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        self.policy_net = DRQN(
            self.n_states,
            self.n_hidden,
            self.n_actions,
            self.n_layers,
            bidirectional=config.bidirectional,
        ).to(self.device)
        self.target_net = DRQN(
            self.n_states,
            self.n_hidden,
            self.n_actions,
            self.n_layers,
            bidirectional=config.bidirectional,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True
        )  # amsgrad is a variant of Adam with guaranteed convergence

    def evaluate(  # TODO fix red error
        self,
        env,
        episodes,
        seed=0,
        policy_net_state_dict: dict = {},
        normalization: tuple = (),
    ):
        """Evaluate the agent on the vehicle tracking environment for a number of episodes.
        If a policy_net_state_dict is provided, the agent will use the weights from that
        for its policy network.

        Parameters
        ----------
        env : VehicleTracking
            The vehicle tracking environment to evaluate the agent on.
        episodes : int
            The number of episodes to evaluate the agent for.
        seed : int, optional
            The seed to use for the random number generator, by default 0.
        policy_net_state_dict : dict, optional
            The state dictionary of the policy network, by default {}.

        Returns
        -------
        tuple[np.ndarray, dict]
            The returns for each episode, and a dictionary of additional information."""
        if policy_net_state_dict:
            self.policy_net.load_state_dict(policy_net_state_dict)
        if self.normalize:
            if not normalization:
                raise ValueError(
                    "Normalization tuple must be provided for this config."
                )
            self.running_mean_std = RunningMeanStd(shape=(2,))
            self.running_mean_std.mean = normalization[0]
            self.running_mean_std.var = normalization[1]
        self.policy_net.eval()
        returns, info = super().evaluate(
            env,
            episodes,
            seed,
        )
        return returns, {**info, "infeasible": self.infeasible}

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        """Get the MPC action for the given state. Gears are chosen by
        the neural network, while the engine torque and brake force are
        then chosen by the MPC controller. If the gear choice is infeasible,
        the agent will attempt two backup solutions (See backup_1 and backup_2).

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle.

        Returns
        -------
        tuple[float, float, int, dict]
            The engine torque, brake force, and gear chosen by the agent, and an
            info dict containing network state and action."""
        # get gears either from heuristic or from expert mpc for first time step
        infeas_flag = False
        if self.first_timestep:
            nn_state = None
            network_action = None
            if self.expert_mpc:
                expert_sol = self.expert_mpc.solve(
                    {
                        "x_0": state,
                        "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                        "T_e_prev": self.T_e_prev,
                        "gear_prev": self.gear_prev,
                    }
                )
                if not expert_sol.success:
                    raise RuntimeError(
                        "Initial gear choice from expert mpc was infeasible"
                    )
                gear_choice_binary = expert_sol.vals["gear"].full()
                self.gear_choice_explicit = np.argmax(gear_choice_binary, axis=0)
            else:
                gear_choice_binary = self.binary_from_explicit(
                    self.gear_choice_explicit
                )
            self.last_gear_choice_explicit = self.gear_choice_explicit
            self.first_timestep = False
        # get gears from network for non-first time steps
        else:
            T_e, F_b, w_e, x, shifted_gear = self.get_shifted_values_from_sol(
                self.sol, state, self.last_gear_choice_explicit
            )
            nn_state = self.relative_state(
                x,
                T_e,
                F_b,
                w_e,
                shifted_gear,
            )
            gear_choice_binary, network_action = self.get_binary_gear_choice(nn_state)

        self.sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear": gear_choice_binary,
            }
        )
        if not self.sol.success:
            infeas_flag = True
            self.sol, self.gear_choice_explicit = self.backup_1(state)
            if not self.sol.success:
                self.sol, self.gear_choice_explicit = self.backup_2(state)
                if not self.sol.success:
                    if not self.train_flag:
                        raise RuntimeError(
                            "Backup gear solutions were still infeasible."
                        )
                    else:  # TODO  should we put heuristic mpc here?
                        # pass
                        raise RuntimeError(
                            "Backup gear solutions were still infeasible."
                        )

        self.gear = int(self.gear_choice_explicit[0])
        self.last_gear_choice_explicit = self.gear_choice_explicit
        return (
            self.sol.vals["T_e"].full()[0, 0],
            self.sol.vals["F_b"].full()[0, 0],
            self.gear,
            {
                "nn_state": nn_state,
                "network_action": network_action,
                "infeas": infeas_flag,
            },
        )

    def backup_1(self, state: np.ndarray) -> tuple[Solution, np.ndarray]:
        """A backup solution for when the MPC solver fails to find a feasible solution.
        The agent will use the previous gear choice and shift it, appending the last entry.

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle.

        Returns
        -------
        tuple
            The solution to the MPC problem and the explicit gear choice."""
        gear_choice_explicit = np.concatenate(
            (self.last_gear_choice_explicit[1:], [self.last_gear_choice_explicit[-1]])
        )
        gear_choice_binary = self.binary_from_explicit(gear_choice_explicit)
        return (
            self.mpc.solve(
                {
                    "x_0": state,
                    "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    "T_e_prev": self.T_e_prev,
                    "gear": gear_choice_binary,
                }
            ),
            gear_choice_explicit,
        )

    def backup_2(self, state: np.ndarray) -> tuple[Solution, np.ndarray]:
        """A backup solution for when the MPC solver fails to find a feasible solution.
        The agent will use the same gear for all time steps, with the gear determined
        by the velocity of the vehicle.

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle.

        Returns
        -------
        tuple
            The solution to the MPC problem and the explicit gear choice."""
        gear = self.gear_from_velocity(state[1])
        gear_choice_explicit = np.ones((self.N,)) * gear
        gear_choice_binary = self.binary_from_explicit(gear_choice_explicit)
        return (
            self.mpc.solve(
                {
                    "x_0": state,
                    "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    "T_e_prev": self.T_e_prev,
                    "gear": gear_choice_binary,
                }
            ),
            gear_choice_explicit,
        )

    def network_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get the action from the policy network for the given state.
        An epsilon-greedy policy is used for exploration, based on
        self.decay_rate, self.eps_start, and self.steps_done. If exploring,
        the action is chosen randomly.

        Parameters
        ----------
        state : torch.Tensor
            The state of the vehicle.

        Returns
        -------
        torch.Tensor
            The action chosen by the policy network (or random action)."""
        eps_threshold = self.eps_start * np.exp(-self.decay_rate * self.steps_done)
        sample = self.np_random.random()
        # dont explore if greater than threshold or if not training
        if sample > eps_threshold or not self.train_flag:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax(2)

        actions = torch.tensor(
            [self.np_random.integers(0, self.n_actions) for _ in range(state.shape[1])],
            device=self.device,
            dtype=torch.long,  # long -> integers
        )
        return actions.unsqueeze(0)

    def on_train_start(self):
        self.steps_done = 0
        self.infeasible = []

    def on_validation_start(self):
        self.T_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.F_b = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.w_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.x = torch.empty((2, self.N), device=self.device, dtype=torch.float32)
        return super().on_validation_start()

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.first_timestep = True
        self.infeasible.append([])
        # store a default gear based on velocity when episode starts
        self.gear = self.gear_from_velocity(state[1])
        self.gear_choice_explicit: np.ndarray = np.ones((self.N,)) * self.gear
        return super().on_episode_start(state, env)

    def on_env_step(self, env, episode, timestep, info):
        self.steps_done += 1
        if self.normalize:
            diff = info["x"] - info["x_ref"]
            self.running_mean_std.update(
                diff.T
            )  # transpose needed as the mean is taken over axis 0
        if "infeas" in info:
            self.infeasible[-1].append(info["infeas"])
        return super().on_env_step(env, episode, timestep, info)

    def relative_state(
        self,
        x: torch.Tensor,
        T_e: torch.Tensor,
        F_b: torch.Tensor,
        w_e: torch.Tensor,
        gear: torch.Tensor,
    ) -> torch.Tensor:
        d_rel = x[:, [0]] - torch.from_numpy(self.x_ref_predicition[:-1, 0]).to(
            self.device
        )
        v_rel = x[:, [1]] - torch.from_numpy(self.x_ref_predicition[:-1, 1]).to(
            self.device
        )
        if self.normalize and self.steps_done > 1:
            d_rel = (d_rel - self.running_mean_std.mean[0]) / (
                self.running_mean_std.var[0] + 1e-6
            )
            v_rel = (v_rel - self.running_mean_std.mean[1]) / (
                self.running_mean_std.var[1] + 1e-6
            )
            T_e = (T_e - Vehicle.T_e_idle) / (Vehicle.T_e_max - Vehicle.T_e_idle)
            F_b = F_b / Vehicle.F_b_max  # min is zero
            w_e = (w_e - Vehicle.w_e_idle) / (Vehicle.w_e_max - Vehicle.w_e_idle)
            gear = gear / 5
        v_norm = (x[:, [1]] - Vehicle.v_min) / (Vehicle.v_max - Vehicle.v_min)
        v_target_norm = (
            torch.from_numpy(self.x_ref_predicition[:-1, 1]).to(self.device)
            - Vehicle.v_min
        ) / (Vehicle.v_max - Vehicle.v_min)
        return (
            torch.cat((d_rel, v_rel, v_norm, v_target_norm, T_e, F_b, w_e, gear), dim=1)
            .unsqueeze(0)
            .to(torch.float32)
        )

    def binary_from_explicit(self, explicit: np.ndarray) -> np.ndarray:
        """Converts the explicit gear choice to a one-hot binary representation.

        Parameters
        ----------
        explicit : np.ndarray
            The explicit gear choice, (N,) with integers 0-5

        Returns
        -------
        np.ndarray
            The binary gear choice, (6, N) with 1s at the explicit gear choice"""
        binary = np.zeros((self.n_gears, self.N))
        binary[explicit.astype(int), np.arange(self.N)] = 1
        return binary

    def get_binary_gear_choice(
        self, nn_state: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor]:
        network_action = self.network_action(nn_state)
        if self.n_actions == 3:
            gear_shift = (network_action - 1).cpu().numpy()
            self.gear_choice_explicit = np.array(
                [self.gear + np.sum(gear_shift[:, : i + 1]) for i in range(self.N)]
            )
            self.gear_choice_explicit = np.clip(self.gear_choice_explicit, 0, 5)
        elif self.n_actions == 6:
            self.gear_choice_explicit = network_action.cpu().numpy().squeeze()
        return self.binary_from_explicit(self.gear_choice_explicit), network_action

    def get_shifted_values_from_sol(
        self, sol, state, gear_choice_explicit
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = sol.vals["x"].full().T
        T_e = sol.vals["T_e"].full().T
        F_b = sol.vals["F_b"].full().T
        w_e = sol.vals["w_e"].full().T
        x = torch.tensor(
            np.concatenate((state.T, x[2:])), dtype=torch.float32, device=self.device
        )
        T_e = torch.tensor(
            np.concatenate((T_e[1:], T_e[[-1]])),
            dtype=torch.float32,
            device=self.device,
        )
        F_b = torch.tensor(
            np.concatenate((F_b[1:], F_b[[-1]])),
            dtype=torch.float32,
            device=self.device,
        )
        w_e = torch.tensor(
            np.concatenate((w_e[1:], w_e[[-1]])),
            dtype=torch.float32,
            device=self.device,
        )
        gear = (
            torch.from_numpy(
                np.concatenate((gear_choice_explicit[1:], gear_choice_explicit[[-1]]))
            )
            .unsqueeze(1)
            .to(self.device)
        )
        return T_e, F_b, w_e, x, gear


class SupervisedLearningAgent(LearningAgent):

    def train(
        self,
        nn_inputs: torch.Tensor,
        nn_targets: torch.Tensor,
        train_epochs: int = 100,
        save_freq: int = 100,
        save_path: str = "",
    ):
        # TODO add docstring and outputs
        num_eps = nn_inputs.shape[0]

        self.policy_net.to(self.device)
        self.policy_net.train()  # set model to training mode
        nn_targets = torch.argmax(nn_targets, 3)
        s_train_tensor = torch.tensor(
            nn_inputs.reshape(-1, nn_inputs.shape[2], nn_inputs.shape[3]),
            dtype=torch.float32,
        ).to(self.device)
        a_train_tensor = torch.tensor(
            nn_targets.reshape(-1, nn_targets.shape[2]), dtype=torch.long
        ).to(self.device)
        dataset = TensorDataset(s_train_tensor, a_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_history = np.empty(train_epochs, dtype=float)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(train_epochs):
            if epoch % save_freq == 0:
                torch.save(
                    self.policy_net.state_dict(),
                    f"{save_path}policy_net_ep_{num_eps}_epoch_{epoch}.pth",
                )
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.policy_net(inputs)
                # Compute loss
                loss = criterion(outputs.transpose(1, 2), labels)
                # Backward pass
                loss.backward()
                # Update weights
                self.optimizer.step()
                running_loss += loss.item()
            loss_history[epoch] = running_loss
            print(f"Epoch [{epoch+1}/{train_epochs}], Loss: {running_loss}")
        torch.save(  # TODO redo saving nicely
            self.policy_net.state_dict(),
            f"{save_path}policy_net_ep_{num_eps}_epoch_{train_epochs}.pth",
        )
        return running_loss, loss_history

    def generate_supervised_data(
        self,
        env: VehicleTracking,
        episodes: int,
        ep_len: int,
        mpc: HybridTrackingMpc,
        save_path: str,
        save_freq: int = 1000,
        seed: int = 0,
    ) -> None:
        # TODO docstring
        # TODO can this more or less use the evaluate function?
        og_seed = seed
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))

        # self.reset()
        nn_inputs = torch.empty(
            (episodes, ep_len - 1, self.N, self.n_states), dtype=torch.float32
        )  # -1 because the first state is not used
        nn_targets_shift = torch.empty(
            (episodes, ep_len - 1, self.N, 3), dtype=torch.float32
        )  # indicate data type
        nn_targets_explicit = torch.empty(
            (episodes, ep_len - 1, self.N, 6), dtype=torch.float32
        )  # indicate data type
        self.on_validation_start()
        for episode, seed in zip(range(episodes), seeds):
            if episode % save_freq == 0 and episode != 0:
                torch.save(
                    {
                        "inputs": nn_inputs[:episode],
                        "targets_explicit": nn_targets_explicit[:episode],
                        "targets_shift": nn_targets_shift[:episode],
                    },
                    f"{save_path}_nn_data_{episode}_seed_{og_seed}.pth",
                )
            print(f"Supervised data: Episode {episode}")
            state, _ = env.reset(seed=seed)
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(state, env)

            while not (truncated or terminated):
                sol = mpc.solve(
                    {
                        "x_0": state,
                        "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                        "T_e_prev": self.T_e_prev,
                        "gear_prev": self.gear_prev,
                    }
                )
                if not sol.success:
                    raise ValueError("MPC failed to solve")
                if timestep != 0:
                    # shift command
                    optimal_gears = np.insert(
                        np.argmax(sol.vals["gear"].full(), 0), 0, self.gear
                    )
                    gear_shift = optimal_gears[1:] - optimal_gears[:-1]
                    action = torch.zeros((self.N, 3), dtype=torch.float32)
                    action[range(self.N), gear_shift + 1] = 1
                    nn_targets_shift[episode, timestep - 1] = action
                    # absolute gear command
                    optimal_gears = np.argmax(sol.vals["gear"].full(), 0)
                    action = torch.zeros((self.N, 6), dtype=torch.float32)
                    action[range(self.N), optimal_gears] = 1
                    nn_targets_explicit[episode, timestep - 1] = action

                    nn_inputs[episode, timestep - 1] = nn_state

                self.gear = int(np.argmax(sol.vals["gear"].full(), 0)[0])
                action = (
                    sol.vals["T_e"].full()[0, 0],
                    sol.vals["F_b"].full()[0, 0],
                    self.gear,
                )
                state, reward, truncated, terminated, step_info = env.step(action)
                self.on_env_step(env, episode, timestep, step_info)

                T_e, F_b, w_e, x, shifted_gear = self.get_shifted_values_from_sol(
                    sol, state, np.argmax(sol.vals["gear"].full(), 0)
                )
                nn_state = self.relative_state(x, T_e, F_b, w_e, shifted_gear)

                timestep += 1
                # self.on_timestep_end()
            # self.on_episode_end()

        # self.on_validation_end()
        torch.save(
            {
                "inputs": nn_inputs,
                "targets_explicit": nn_targets_explicit,
                "targets_shift": nn_targets_shift,
            },
            f"{save_path}_nn_data_{episode}_seed_{og_seed}.pth",
        )
        return


class DQNAgent(LearningAgent):
    # TODO docstring

    cost: list[list[float]] = []  # store the incurred cost: env.reward + penalties

    def __init__(self, mpc, np_random, config):
        self.train_flag = False

        # RL hyperparameters
        self.gamma = config.gamma
        self.tau = config.tau

        # exploration
        self.eps_start = config.eps_start
        self.decay_rate = 0  # gets decided at the start of training

        # penalties
        self.clip_pen = config.clip_pen
        self.infeas_pen = config.infeas_pen

        # memory
        self.memory_size = config.memory_size
        self.memory = ReplayMemory(self.memory_size)

        # max gradient value to avoid exploding gradients
        self.max_grad = config.max_grad
        super().__init__(mpc, np_random, config)

    def train(
        self,
        env: VehicleTracking,
        episodes: int,
        seed: int = 0,
        save_freq: int = 0,
        save_path: str = "",
        exp_zero_steps: int = 0,
        init_state_dict: dict = {},
        max_learning_steps: int = np.inf,
    ) -> tuple[np.ndarray, dict]:
        """Train the policy on the environment using deep Q-learning.

        Parameters
        ----------
        env : VehicleTracking
            The vehicle tracking environment on which the policy is trained.
        episodes : int
            The number of episodes to train the policy for.
        seed : int, optional
            The seed for the random number generator, by default 0.
        save_freq : int, optional
            The step frequency at which to save the policy, by default 0.
        save_path : str, optional
            The path to the folder where the data and models are saved.
        exp_zero_steps : int, optional
            The number of steps at which the exploration rate is desired to be
            approximately zero (1e-3), by default 0 (no exploration).
        init_state_dict : dict, optional
            The initial state dictionary for the Q and target network, by default {}
            in which case a randomized start is used.
        max_learning_steps : int, optional
            The maximum total number of learning steps across all episodes, after
            which the training terminates."""
        if self.normalize:
            self.running_mean_std = RunningMeanStd(shape=(2,))

        if init_state_dict:
            self.policy_net.load_state_dict(init_state_dict)
            self.target_net.load_state_dict(init_state_dict)

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        returns = np.zeros(episodes)
        self.decay_rate = np.log(self.eps_start / 1e-3) / exp_zero_steps
        total_steps = 0

        self.on_train_start()
        for episode, seed in zip(range(episodes), seeds):
            print(f"Train: Episode {episode}")
            state, step_info = env.reset(seed=seed)
            self.on_episode_start(state, env)
            timestep = 0
            terminated = truncated = False

            while not (terminated or truncated):
                T_e, F_b, gear, action_info = self.get_action(state)
                penalty = self.infeas_pen if action_info["infeas"] else 0

                state, reward, truncated, terminated, step_info = env.step(
                    (T_e, F_b, gear)
                )
                returns[episode] += reward
                self.on_env_step(env, episode, timestep, step_info | action_info)

                T_e, F_b, w_e, x, shifted_gear = self.get_shifted_values_from_sol(
                    self.sol, state, self.last_gear_choice_explicit
                )
                nn_next_state = self.relative_state(
                    x,
                    T_e,
                    F_b,
                    w_e,
                    shifted_gear,
                )

                # Store the transition in memory
                if action_info["nn_state"] is not None:  # is None for first time step
                    self.memory.push(
                        action_info["nn_state"],
                        action_info["network_action"],
                        nn_next_state,
                        torch.tensor([reward + penalty]),
                    )

                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                self.on_timestep_end(reward + penalty)
                timestep += 1
                if timestep % 100 == 0:
                    print(f"Episode {episode}: Step {timestep}")
                total_steps += 1
                if total_steps >= max_learning_steps:
                    break
                if save_freq and total_steps % save_freq == 0:
                    self.save(env=env, step=total_steps, path=save_path)

        print("Training complete")
        self.save(env=env, step=total_steps, path=save_path)
        info = {
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
            "cost": self.cost,
            "infeasible": self.infeasible,
        }
        if self.normalize:
            info["normalization"] = (
                self.running_mean_std.mean,
                self.running_mean_std.var,
            )
        return returns, info

    def optimize_model(self):
        """Apply a policy update based on the current memory buffer"""
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size, self.np_random)
        batch = Transition(
            *zip(*transitions)
        )  # convert batch of transitions to transition of batches

        non_final_mask = torch.tensor(  # mask for transitions that are not final (where the simulation ended)
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = -torch.cat(batch.reward)  # negate cost to become reward

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(
            2, action_batch.unsqueeze(2)
        )

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, self.N, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(2).values
            )

        # Compute the expected Q values, extend the reward to the length of the sequence
        reward_batch = reward_batch.unsqueeze(1).expand(-1, self.N).to(self.device)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute Huber loss``
        criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values[:,0,:], expected_state_action_values.unsqueeze(-1))
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(-1)
        )
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.max_grad)
        self.optimizer.step()

    def save(self, env: VehicleTracking, step: int, path: str = ""):
        torch.save(self.policy_net.state_dict(), path + f"/policy_net_step_{step}.pth")
        torch.save(self.target_net.state_dict(), path + f"/target_net_step_{step}.pth")
        info = {
            "cost": self.cost,
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
            "infeasible": self.infeasible,
            "R": list(env.rewards) + [np.asarray(env.ep_rewards)],
            "X": list(env.observations) + [np.asarray(env.ep_observations)],
            "U": list(env.actions) + [np.asarray(env.ep_actions)],
        }
        if self.normalize:
            info["normalization"] = (
                self.running_mean_std.mean,
                self.running_mean_std.var,
            )
        with open(path + f"/data_step_{step}.pkl", "wb") as f:
            pickle.dump(
                info,
                f,
            )

    def on_timestep_end(self, cost: float):
        self.cost[-1].append(cost)

    def on_train_start(self):
        self.train_flag = True
        self.cost = []
        super().on_train_start()

    def on_validation_start(self):
        self.train_flag = False
        return super().on_validation_start()

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.cost.append([])
        return super().on_episode_start(state, env)


class DistributedAgent(Agent):
    num_vehicles: int = 3

    def get_action(self, state):
        return (
            np.random.uniform(15, 300, self.num_vehicles),
            np.random.uniform(0, 9000, self.num_vehicles),
            [np.random.randint(0, 6) for _ in range(self.num_vehicles)],
            {},
        )

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.fuel.append([])
        self.engine_torque.append([])
        self.engine_speed.append([])
        self.x_ref.append(np.empty((0, 2, 1)))
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()
        self.T_e_prev = [Vehicle.T_e_idle for _ in range(self.num_vehicles)]
        gear = [
            self.gear_from_velocity(state[1, i].item())
            for i in range(self.num_vehicles)
        ]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(self.num_vehicles)]
        for i, g in enumerate(gear):
            self.gear_prev[i][g] = 1

    def on_env_step(
        self, env: VehicleTracking, episode: int, timestep: int, info: dict
    ):
        self.fuel[episode].append(info["fuel"])
        self.engine_torque[episode].append(info["T_e"])
        self.engine_speed[episode].append(info["w_e"])
        self.x_ref[episode] = np.concatenate(
            (self.x_ref[episode], info["x_ref"].reshape(1, 2, 1))
        )
        self.x_ref_predicition = env.unwrapped.get_x_ref_prediction()
        self.T_e_prev = info["T_e"]
        gear = info["gear"]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(self.num_vehicles)]
        for i, g in enumerate(gear):
            self.gear_prev[i][g] = 1

    def clip_action(self, action):
        for i in range(self.num_vehicles):
            if np.abs(np.argmax(self.gear_prev[i]) - action[2][i]) > 1:
                action[2][i] = np.clip(
                    action[2][i],
                    np.argmax(self.gear_prev[i]) - 1,
                    np.argmax(self.gear_prev[i]) + 1,
                )
        return action
