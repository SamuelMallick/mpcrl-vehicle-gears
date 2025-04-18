import pickle
from typing import Literal
import numpy as np
import torch
import torch.optim as optim
from agents.learning_agent import LearningAgent
from env import VehicleTracking
from network import DRQN, ReplayMemory, Transition
from utils.running_mean_std import RunningMeanStd
import torch.nn as nn


class DQNAgent(LearningAgent):
    """An agent that uses a learning-based policy to select the gear-shift
    schedule. An NLP-based MPC controller then solves for the continuous
    variables, given the gear-shift schedule. To train the policy the DQN
    RL algorithm is used.

    Parameters
    ----------
    mpc : FixedGearMPC
        The MPC controller used to solve the continuous variables.
    np_random : np.random.Generator
        The random number generator.
    config : ConfigDefault
        A configuration file containing initialization parameters for the
        learning component of the agent.
    multi_starts : int, optional
        The number of multi-starts to use. The default is 1."""

    def __init__(self, mpc, np_random, config, multi_starts=1):
        super().__init__(mpc, np_random, config, multi_starts=multi_starts)

        # RL hyperparameters
        self.gamma = config.gamma
        self.tau = config.tau

        # exploration
        self.eps_start = config.eps_start
        self.decay_rate = 0  # gets decided at the start of training

        # penalties
        self.infeas_pen = config.infeas_pen
        self.rl_reward = config.rl_reward

        # memory
        self.memory_size = config.memory_size
        self.memory = ReplayMemory(self.memory_size)

        # max gradient value to avoid exploding gradients
        self.max_grad = config.max_grad

        self.target_net = DRQN(
            config.n_states,
            config.n_hidden,
            config.n_actions,
            config.n_layers,
            bidirectional=config.bidirectional,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )  # amsgrad is a variant of Adam with guaranteed convergence

    def train(
        self,
        env: VehicleTracking,
        episodes: int,
        seed: int = 0,
        save_freq: int = 0,
        save_path: str = "",
        exp_zero_steps: int = 0,
        use_heuristic: bool = False,
        heursitic_gear_priorities: list[Literal["low", "high", "mid"]] = ["low"],
        init_state_dict: dict = {},
        init_normalization: tuple = (),
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
        use_heuristic : bool
            If True, a heuristic MPC will also solve the problem at each timestep,
            and the best solution between the heuristic and the neural network will be used.
        heursitic_gear_priorities : list[Literal["low", "high", "mid"]]
            If use_heuristic is True: For each entry in the list an MPC will also be solved
            using a fixed gear schedule determined by the heuristic (see also heuristic_2_agent.py).
        init_state_dict : dict, optional
            The initial state dictionary for the Q and target network, by default {}
            in which case a randomized start is used.
        max_learning_steps : int, optional
            The maximum total number of learning steps across all episodes, after
            which the training terminates."""
        self.use_heuristic = use_heuristic
        self.heursitic_gear_priorities = heursitic_gear_priorities

        if self.normalize:
            self.running_mean_std = RunningMeanStd(shape=(2,))
            if init_normalization:
                self.running_mean_std.mean = init_normalization[0]
                self.running_mean_std.var = init_normalization[1]

        if init_state_dict:
            self.policy_net.load_state_dict(init_state_dict)
            self.target_net.load_state_dict(init_state_dict)

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        returns = np.zeros(episodes)
        self.decay_rate = (
            np.log(self.eps_start / 1e-3) / exp_zero_steps if self.eps_start > 0 else 0
        )

        self.on_train_start()
        for episode, seed in zip(range(episodes), seeds):
            print(f"Train: Episode {episode}")
            state, step_info = env.reset(seed=seed)
            self.on_episode_start(state, env)
            timestep = 0
            terminated = truncated = False

            while not (terminated or truncated):
                *action, action_info = self.get_action(state)
                action = self.clip_action(action)
                T_e, F_b, gear = action
                penalty = self.infeas_pen if action_info["infeasible"] else 0
                penalty += self.rl_reward if not action_info["heuristic"] else 0

                state, reward, truncated, terminated, step_info = env.step(
                    (T_e, F_b, gear)
                )
                returns[episode] += reward
                self.on_env_step(env, episode, timestep, step_info | action_info)

                nn_next_state = self.relative_state(
                    *self.get_nn_inputs_from_sol(
                        self.prev_sol, state, self.prev_gear_choice_explicit
                    ),
                    self.x_ref_predicition[:-1, :].reshape(-1, 2),
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

                if self.steps_done >= max_learning_steps:
                    self.save(env=env, step=self.steps_done, path=save_path)
                    return returns, {}
                if save_freq and self.steps_done % save_freq == 0:
                    self.save(env=env, step=self.steps_done, path=save_path)

        self.save(env=env, step=self.steps_done, path=save_path)
        return returns, {}

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
        next_state_values = torch.zeros(
            self.batch_size, self.mpc.prediction_horizon, device=self.device
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(2).values
            )

        # Compute the expected Q values, extend the reward to the length of the sequence
        reward_batch = (
            reward_batch.unsqueeze(1)
            .expand(-1, self.mpc.prediction_horizon)
            .to(self.device)
        )
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

    def on_train_start(self):
        self.policy_net.train()
        self.target_net.train()
        self.cost = []
        self.steps_done = 0
        self.heuristic_flags: list[list[bool]] = []
        self.steps_done = 0

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.cost.append([])
        return super().on_episode_start(state, env)

    def on_timestep_end(self, cost: float):
        self.cost[-1].append(cost)

    def save(self, env: VehicleTracking, step: int, path: str = ""):
        torch.save(self.policy_net.state_dict(), path + f"/policy_net_step_{step}.pth")
        torch.save(self.target_net.state_dict(), path + f"/target_net_step_{step}.pth")
        info = {
            "cost": self.cost,
            "fuel": list(env.fuel_consumption) + [np.asarray(env.ep_fuel_consumption)],
            "T_e": list(env.engine_torque) + [np.asarray(env.ep_engine_torque)],
            "w_e": list(env.engine_speed) + [np.asarray(env.ep_engine_speed)],
            "x_ref": list(env.reference_trajectory)
            + [np.asarray(env.ep_reference_trajectory)],
            "heuristic": self.heuristic_flags,
            "infeasible": list(self.infeasible_flags),
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
