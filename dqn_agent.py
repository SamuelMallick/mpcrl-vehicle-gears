from collections import deque, namedtuple
import time

from csnlp import Solution
from agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from env import VehicleTracking
from mpc import HybridTrackingMpc
from vehicle import Vehicle
import pickle
from config_files.base import ConfigDefault

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently.

    Parameters
    ----------
    capacity : int
        The maximum number of transitions that can be stored in the memory."""

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(
        self, batch_size: int, np_random: np.random.Generator
    ) -> list[Transition]:
        index_samples = np_random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in index_samples]

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    """A deep recurrent Q-network (DRQN) that maps state sequences to Q-values for
    gear shifts. The architecture is a (bidirectional) RNN followed by a fully connected
    layer.

    Parameters
    ----------
    input_size : int
        The size of the input state vector.
    hidden_size : int
        The number of features in the hidden state of the RNN.
    num_actions : int, optional
        The number of actions that can be taken, by default 3.
        Three actions : downshift, no shift, upshift.
    num_layers : int, optional
        The number of recurrent layers, by default 1."""

    def __init__(
        self, input_size, hidden_size, num_actions=3, num_layers=1, bidirectional=False
    ):
        super(DRQN, self).__init__()
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, num_actions
        )
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        drqn_out, _ = self.rnn(x)
        q_values = self.fc(drqn_out)
        return q_values


class DQNAgent(Agent):
    """An RL agent that use deep Q-learning to learn a policy for gear shifting.

    Parameters
    ----------
    mpc : HybridTrackingMpc
        The MPC controller used to simulate the vehicle. Gear choice made by the
        agent are passed to this MPC controller, which then decides on the engine
        torque and brake force.
    np_random : np.random.Generator
        A random number generator used for exploration and sampling.
    config : Config
        A configuration object that specifies hyperparameters and architecture of the
        neural network and agent, by default None. See config_files/base.py for the
        configuration details
    """

    cost: list[list[float]] = []  # store the incurred cost: env.reward + penalties
    infeasible: list[list[float]] = []  # store if the gear choice was infeasible

    def __init__(
        self,
        mpc: HybridTrackingMpc,
        np_random: np.random.Generator,
        config: ConfigDefault,
    ):
        super().__init__(mpc)
        self.train_flag = False
        self.first_timestep: bool = True  # gets set to True in on_episode_start
        self.np_random = np_random
        self.n_gears = 6

        self.expert_mpc = config.expert_mpc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("Using GPU")

        # hyperparameters
        self.gamma = config.gamma
        self.learning_rate = config.learning_rate
        self.tau = config.tau

        # archticeture
        self.n_states = config.n_states
        self.n_hidden = config.n_hidden
        self.n_actions = config.n_actions
        self.n_layers = config.n_layers
        self.N = config.N
        self.normalize = config.normalize

        # exploration
        self.eps_start = config.eps_start
        self.steps_done = 0
        self.decay_rate = 0  # gets decided at the start of training

        # penalties
        self.clip_pen = config.clip_pen
        self.infeas_pen = config.infeas_pen

        # memory
        self.memory_size = config.memory_size
        self.memory = ReplayMemory(self.memory_size)
        self.batch_size = config.batch_size

        # max gradient value to avoid exploding gradients
        self.max_grad = config.max_grad

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

    def train_supervised(
        self,
        nn_inputs: torch.Tensor,
        nn_targets: torch.Tensor,
        batch_size: int = 128,
        train_epochs: int = 100,
        save_freq: int = 100,
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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_history = np.empty(train_epochs, dtype=float)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(train_epochs):
            if epoch % save_freq == 0:
                torch.save(
                    self.policy_net.state_dict(),
                    f"policy_net_ep_{num_eps}_epoch_{epoch}.pth",
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
        torch.save(
            self.policy_net.state_dict(),
            f"policy_net_ep_{num_eps}_epoch_{train_epochs}.pth",
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
                    f"{save_path}_nn_data_{episode}.pth",
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

                self.T_e, self.F_b, self.w_e, self.x = self.get_vals_from_sol(sol)
                nn_state = self.relative_state(
                    self.x,
                    self.T_e,
                    self.F_b,
                    self.w_e,
                    torch.from_numpy(np.argmax(sol.vals["gear"].full(), 0))
                    .unsqueeze(1)
                    .to(self.device),
                )
                self.gear = int(np.argmax(sol.vals["gear"].full(), 0)[0])
                action = (
                    sol.vals["T_e"].full()[0, 0],
                    sol.vals["F_b"].full()[0, 0],
                    self.gear,
                )
                state, reward, truncated, terminated, step_info = env.step(action)
                self.on_env_step(env, episode, timestep, step_info)

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
            f"{save_path}_nn_data_{episode}.pth",
        )
        return

    def evaluate(self, env, episodes, seed=0, policy_net_state_dict: dict = {}):
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
        self.policy_net.eval()
        returns, info = super().evaluate(env, episodes, seed)
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
            nn_state_prev = None
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
            self.first_timestep = False
        # get gears from network for non-first time steps
        else:
            nn_state_prev = self.nn_state
            gear_choice_binary, network_action = self.get_binary_gear_choice(self.nn_state)

        sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear": gear_choice_binary,
            }
        )
        if not sol.success:
            infeas_flag = True
            sol, self.gear_choice_explicit = self.backup_1(state)
            if not sol.success:
                sol, self.gear_choice_explicit = self.backup_2(state)
                if not sol.success:
                    if not self.train_flag:
                        raise RuntimeError(
                            "Backup gear solutions were still infeasible."
                        )
                    else:  # TODO  should we put heuristic mpc here?
                        # pass
                        raise RuntimeError(
                            "Backup gear solutions were still infeasible."
                        )
                    # expert_sol = self.expert_mpc.solve(
                    #     {
                    #         "x_0": state,
                    #         "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    #         "T_e_prev": self.T_e_prev,
                    #         "gear_prev": self.gear_prev,
                    #     }
                    # )
        self.T_e, self.F_b, self.w_e, self.x = self.get_vals_from_sol(sol)
        self.nn_state = self.relative_state(
            self.x,
            self.T_e,
            self.F_b,
            self.w_e,
            torch.from_numpy(self.gear_choice_explicit).unsqueeze(1).to(self.device),
        )

        self.gear = int(self.gear_choice_explicit[0])
        self.last_gear_choice_explicit = self.gear_choice_explicit
        return (
            sol.vals["T_e"].full()[0, 0],
            sol.vals["F_b"].full()[0, 0],
            self.gear,
            {
                "nn_state": nn_state_prev,
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

    def train(
        self,
        env: VehicleTracking,
        episodes: int,
        ep_len: int,
        seed: int = 0,
        save_freq: int = 0,
        save_path: str = "",
        exp_zero_steps: int = 0,
    ) -> tuple[np.ndarray, dict]:
        """Train the policy on the environment using deep Q-learning.

        Parameters
        ----------
        env : VehicleTracking
            The vehicle tracking environment on which the policy is trained.
        episodes : int
            The number of episodes to train the policy for.
        ep_len : int
            The number of time steps in each episode.
        seed : int, optional
            The seed for the random number generator, by default 0.
        save_freq : int, optional
            The episode frequency at which to save the policy, by default 0.
        save_path : str, optional
            The path to the folder where the data and models are saved.
        exp_zero_steps : int, optional
            The number of steps at which the exploration rate is desired to be
            approximately zero (1e-3), by default 0 (no exploration)."""
        if self.normalize:
            self.position_error = np.zeros((episodes, ep_len))
            self.velocity_error = np.zeros((episodes, ep_len))

        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        returns = np.zeros(episodes)
        self.decay_rate = np.log(self.eps_start / 1e-3) / exp_zero_steps

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

                nn_next_state = self.relative_state(
                    self.x,
                    self.T_e,
                    self.F_b,
                    self.w_e,
                    torch.from_numpy(self.gear_choice_explicit)
                    .unsqueeze(1)
                    .to(self.device),
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

            if save_freq and episode % save_freq == 0:
                self.save(env=env, ep=episode, path=save_path)

        print("Training complete")
        self.save(env=env, ep=episode, path=save_path)
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
                np.mean(self.position_error),
                np.std(self.position_error),
                np.mean(self.velocity_error),
                np.std(self.velocity_error),
            )
        return returns, info

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

    def on_train_start(self):
        self.train_flag = True
        self.steps_done = 0
        self.cost = []
        self.infeasible = []

    def on_validation_start(self):
        self.train_flag = False
        self.T_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.F_b = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.w_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.x = torch.empty((2, self.N), device=self.device, dtype=torch.float32)
        return super().on_validation_start()

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.first_timestep = True
        self.cost.append([])
        self.infeasible.append([])
        # store a default gear based on velocity when episode starts
        self.gear = self.gear_from_velocity(state[1])
        self.gear_choice_explicit: np.ndarray = np.ones((self.N,)) * self.gear
        return super().on_episode_start(state, env)

    def on_timestep_end(self, cost: float):
        self.cost[-1].append(cost)

    def on_env_step(self, env, episode, timestep, info):
        self.steps_done += 1
        if self.normalize:
            self.position_error[episode, timestep] = (
                self.x[0, 0].cpu() - self.x_ref_predicition[0, 0]
            ).item()
            self.velocity_error[episode, timestep] = (
                self.x[0, 1].cpu() - self.x_ref_predicition[0, 1]
            ).item()
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
            d_rel = (
                d_rel - np.mean(self.position_error.flatten()[: self.steps_done])
            ) / (np.std(self.position_error.flatten()[: self.steps_done]) + 1e-6)
            v_rel = (
                v_rel - np.mean(self.velocity_error.flatten()[: self.steps_done])
            ) / (np.std(self.velocity_error.flatten()[: self.steps_done]) + 1e-6)
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

    def save(self, env: VehicleTracking, ep: int, path: str = ""):
        torch.save(self.policy_net.state_dict(), path + f"/policy_net_ep_{ep}.pth")
        torch.save(self.target_net.state_dict(), path + f"/target_net_ep_{ep}.pth")
        info = {
            "cost": self.cost,
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
            "infeasible": self.infeasible,
            "R": list(env.rewards),
            "X": list(env.observations),
            "U": list(env.actions),
        }
        if self.normalize:
            info["normalization"] = (
                np.mean(self.position_error),
                np.std(self.position_error),
                np.mean(self.velocity_error),
                np.std(self.velocity),
            )
        with open(path + f"/data_ep_{ep}.pkl", "wb") as f:
            pickle.dump(
                info,
                f,
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

    def get_vals_from_sol(
        self, sol
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T_e = torch.tensor(
            sol.vals["T_e"].full().T, dtype=torch.float32, device=self.device
        )
        F_b = torch.tensor(
            sol.vals["F_b"].full().T, dtype=torch.float32, device=self.device
        )
        w_e = torch.tensor(
            sol.vals["w_e"].full().T, dtype=torch.float32, device=self.device
        )
        x = torch.tensor(
            sol.vals["x"][:, :-1].full().T, dtype=torch.float32, device=self.device
        )
        return T_e, F_b, w_e, x
