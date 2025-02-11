from collections import deque, namedtuple
import random
from agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import VehicleTracking
from vehicle import Vehicle
import pickle

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    # TODO docstring

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(
        self, batch_size: int, np_random: np.random.Generator
    ):  # TODO add return type
        return random.sample(
            self.memory, batch_size
        )  # TODO make this a seeded operation
        # return np_random.choice(list(self.memory), batch_size, replace=False)

    def __len__(self):
        return len(self.memory)


class DRQN(nn.Module):
    # TODO docstring

    def __init__(
        self, input_size, hidden_size, num_actions=3, num_layers=1
    ):  # three actions : downshift, no shift, upshift
        super(DRQN, self).__init__()
        self.fc = nn.Linear(hidden_size * 2, num_actions)  # fully connected layer
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )  # recurrent layer

    def forward(self, x):
        drqn_out, _ = self.rnn(x)
        q_values = self.fc(drqn_out)
        return q_values


class DQNAgent(Agent):

    cost: list[list[float]] = []

    def __init__(self, mpc, N: int, np_random: np.random.Generator, expert_mpc=None):
        # TODO add docstring
        super().__init__(mpc)
        self.train_flag = True

        self.expert_mpc = expert_mpc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("Using GPU")

        # hyperparameters
        self.gamma = 0.9
        self.learning_rate = 0.001
        self.tau = 0.001

        # archticeture
        self.n_states = 7
        self.n_hidden = 64
        self.n_actions = 3
        self.n_layers = 2
        self.N = N
        self.n_gears = 6

        # exploration
        self.eps_start = 0.99
        self.steps_done = 0
        self.decay_rate = 0

        # penalties
        self.clip_pen = 0
        self.infeas_pen = 1e4

        # memory
        self.memory_size = 100000
        self.memory = ReplayMemory(self.memory_size)
        self.batch_size = 128

        self.np_random = np_random

        seed = np_random.integers(0, 2**32 - 1)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        self.policy_net = DRQN(
            self.n_states, self.n_hidden, self.n_actions, self.n_layers
        ).to(self.device)
        self.target_net = DRQN(
            self.n_states, self.n_hidden, self.n_actions, self.n_layers
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True
        )  # TODO what is amsgrad?

    def evaluate(self, env, episodes, seed=0, policy_net_state_dict: dict = {}):
        if policy_net_state_dict:
            self.policy_net.load_state_dict(policy_net_state_dict)
        return super().evaluate(env, episodes, seed)

    def get_action(self, state: np.ndarray) -> tuple[float, float, int]:
        # TODO add docstring
        if self.first_time_step:
            if self.expert_mpc:
                expert_sol = self.expert_mpc.solve(
                    {
                        "x_0": state,
                        "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                        "T_e_prev": self.T_e_prev,
                        "gear_prev": self.gear_prev,
                    }
                )
                gear_choice_binary = expert_sol.vals["gear"].full()
                self.gear_choice_explicit = np.argmax(gear_choice_binary, axis=0)
            else:
                gear_choice_binary = np.zeros((self.n_gears, self.N))
                for i in range(self.N):  # TODO remove loop
                    gear_choice_binary[int(self.gear_choice_explicit[i]), i] = (
                        1  # TODO is int cast needed?
                    )
            self.first_time_step = False
        else:
            nn_state = self.relative_state(
                self.x,
                self.T_e,
                self.F_b,
                self.w_e,
                torch.from_numpy(self.gear_choice_explicit)
                .unsqueeze(1)
                .to(self.device),
            )
            with torch.no_grad():
                q_values = self.policy_net(nn_state)
                action = q_values.argmax(2)

            gear_shift = action - 1
            gear_shift = gear_shift.cpu().numpy()
            self.gear_choice_explicit = np.array(
                [self.gear + np.sum(gear_shift[:, : i + 1]) for i in range(self.N)]
            )
            self.gear_choice_explicit = np.clip(self.gear_choice_explicit, 0, 5)
            gear_choice_binary = np.zeros((self.n_gears, self.N))
            for i in range(self.N):  # TODO remove loop
                gear_choice_binary[int(self.gear_choice_explicit[i]), i] = (
                    1  # TODO is int cast needed?
                )
        sol = self.mpc.solve(
            {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear": gear_choice_binary,
            }
        )
        if not sol.success:
            sol, self.gear_choice_explicit = self.backup_1(state)
            if not sol.success:
                sol, self.gear_choice_explicit = self.backup_2(state)
                if not sol.success:
                    raise RuntimeError(
                        "Backup gear solutions were still infeasible in eval"
                    )
        self.T_e = torch.tensor(
            sol.vals["T_e"].full().T, dtype=torch.float32, device=self.device
        )
        self.F_b = torch.tensor(
            sol.vals["F_b"].full().T, dtype=torch.float32, device=self.device
        )
        self.w_e = torch.tensor(
            sol.vals["w_e"].full().T, dtype=torch.float32, device=self.device
        )
        self.x = torch.tensor(
            sol.vals["x"][:, :-1].full().T, dtype=torch.float32, device=self.device
        )
        self.gear = int(self.gear_choice_explicit[0])
        self.last_gear_choice_explicit = self.gear_choice_explicit
        return sol.vals["T_e"].full()[0, 0], sol.vals["F_b"].full()[0, 0], self.gear

    def backup_1(self, state: np.ndarray):
        gear_choice_explicit = (
            np.concatenate(  # TODO is there a cleaner way to do this?
                (
                    self.last_gear_choice_explicit[1:],
                    self.last_gear_choice_explicit[[-1]],
                ),
                0,
            )
        )
        gear_choice_binary = np.zeros((self.n_gears, self.N))  # remove loop
        for i in range(self.N):
            gear_choice_binary[int(gear_choice_explicit[i]), i] = 1

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

    def backup_2(self, state: np.ndarray):
        gear = self.gear_from_velocity(state[1])

        # assume all time step have the same gear choice
        gear_choice_explicit = np.ones((self.N,)) * gear
        gear_choice_binary = np.zeros((self.n_gears, self.N))
        # set value for gear choice binary
        for i in range(self.N):  # TODO remove loop
            gear_choice_binary[int(gear_choice_explicit[i]), i] = 1

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
        seed: int = 0,
        save_freq: int = 0,
        save_path: str = "",
        exp_zero_steps: int = 0,
        policy_net_state_dict: dict = {},
        target_net_state_dict: dict = {},
        info_dict: dict = {},
        start_episode: int = 0,
        start_exp_step: int = 0,
    ) -> tuple[np.ndarray, dict]:
        # TODO add docstring
        if policy_net_state_dict:
            self.policy_net.load_state_dict(policy_net_state_dict)
        if target_net_state_dict:
            self.target_net.load_state_dict(target_net_state_dict)
        if info_dict:
            self.cost = info_dict["cost"]
            self.fuel = info_dict["fuel"]
            self.engine_torque = info_dict["T_e"]
            self.engine_speed = info_dict["w_e"]
            self.x_ref = info_dict["x_ref"]
        seeds = map(
            int, np.random.SeedSequence(seed + start_episode).generate_state(episodes)
        )
        returns = np.zeros(episodes)

        self.decay_rate = np.log(1 / 1e-3) / exp_zero_steps

        self.on_train_start()
        self.steps_done = start_exp_step
        for episode, seed in zip(range(start_episode, episodes), seeds):
            print(f"Train: Episode {episode}")
            state, info = env.reset(seed=seed)
            self.on_episode_start(state, env)
            time_step = 0

            # TODO does all this stuff outside of the loop need to be done?
            if self.expert_mpc:
                expert_sol = self.expert_mpc.solve(
                    {
                        "x_0": state,
                        "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                        "T_e_prev": self.T_e_prev,
                        "gear_prev": self.gear_prev,
                    }
                )
                gear_choice_init_binary = expert_sol.vals["gear"].full()
                gear_choice_init_explicit = np.argmax(gear_choice_init_binary, axis=0)
                gear = gear_choice_init_explicit[0]
            else:
                gear = self.gear_from_velocity(state[1])
                gear_choice_init_explicit = np.ones((self.N,)) * gear
                gear_choice_init_binary = np.zeros((self.n_gears, self.N))
                gear_choice_init_binary[gear] = 1

            # solve mpc with fixed gear choice
            sol = self.mpc.solve(
                {
                    "x_0": state,
                    "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    "T_e_prev": self.T_e_prev,
                    "gear": gear_choice_init_binary,
                }
            )
            if not sol.success:
                raise RuntimeError(
                    f"Initial gear choice for episode {episode} was infeasible"
                )

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

            nn_state = self.relative_state(
                x,
                T_e,
                F_b,
                w_e,
                torch.from_numpy(gear_choice_init_explicit)
                .unsqueeze(1)
                .to(self.device),
            )
            self.last_gear_choice_explicit = gear_choice_init_explicit  # TODO remove the need for explicit and binary

            state, reward, truncated, terminated, info = env.step(
                (T_e[0].item(), F_b[0].item(), gear)
            )
            returns[episode] += reward
            self.on_env_step(env, episode, info)
            self.on_timestep_end(reward)
            time_step += 1

            while not (terminated or truncated):
                penalty = 0

                network_action = self.network_action(nn_state)

                gear_shift = network_action - 1
                # NOTE this is different to what Qizhang did, he shifted from previous gear sequence (which was not known by NN)
                gear_shift = gear_shift.cpu().numpy()
                gear_choice_explicit = np.array(
                    [gear + np.sum(gear_shift[:, : i + 1]) for i in range(self.N)]
                )

                # clip gears to be within range
                # penalty += self.clip_pen * np.sum(
                #     gear_choice_explicit < 0
                # ) + self.clip_pen * np.sum(gear_choice_explicit > 5)
                gear_choice_explicit = np.clip(gear_choice_explicit, 0, 5)

                gear_choice_binary = np.zeros((self.n_gears, self.N))
                for i in range(self.N):  # TODO remove loop
                    gear_choice_binary[int(gear_choice_explicit[i]), i] = (
                        1  # TODO is int cast needed?
                    )

                sol = self.mpc.solve(
                    {
                        "x_0": state,
                        "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                        "T_e_prev": self.T_e_prev,
                        "gear": gear_choice_binary,
                    }
                )
                if not sol.success:
                    # backup sol 1: use previous gear choice shifted
                    sol, gear_choice_explicit = self.backup_1(state)

                    if not sol.success:
                        # backup sol 2: use the same gear for all time steps
                        sol, gear_choice_explicit = self.backup_2(state)

                        if not sol.success:
                            # raise RuntimeError(
                            #     "Backup gear solution was still infeasible, reconsider theory. Oh no."
                            # )
                            pass
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
                    sol.vals["x"][:, :-1].full().T,
                    dtype=torch.float32,
                    device=self.device,
                )
                gear = int(gear_choice_explicit[0])

                state, reward, truncated, terminated, info = env.step(
                    (T_e[0].item(), F_b[0].item(), gear)
                )
                returns[episode] += reward
                self.on_env_step(env, episode, info)

                nn_next_state = self.relative_state(
                    x,
                    T_e,
                    F_b,
                    w_e,
                    torch.from_numpy(gear_choice_explicit).unsqueeze(1).to(self.device),
                )

                # Store the transition in memory
                self.memory.push(
                    nn_state,
                    network_action,
                    nn_next_state,
                    torch.tensor([reward + penalty]),
                )

                # Move to the next state
                nn_state = nn_next_state
                self.last_gear_choice_explicit = (
                    gear_choice_explicit  # TODO type hint issue
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
                time_step += 1

            if save_freq and episode % save_freq == 0:
                self.save(env=env, ep=episode, path=save_path)

        print("Training complete")
        self.save(env=env, ep=episode, path=save_path)
        return returns, {
            "fuel": self.fuel,
            "T_e": self.engine_torque,
            "w_e": self.engine_speed,
            "x_ref": self.x_ref,
            "cost": self.cost,
        }

    def network_action(self, state):
        # TODO add docstring
        if self.train_flag:  # epsilon greedy exploration
            eps_threshold = self.eps_start * np.exp(-self.decay_rate * self.steps_done)
        else:
            eps_threshold = 0

        self.steps_done += 1
        sample = self.np_random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                actions = q_values.argmax(2)
        else:  # this random action does not make sense # TODO what does this comment mean?
            actions = torch.tensor(
                [
                    self.np_random.integers(0, self.n_actions)
                    for _ in range(state.shape[1])
                ],
                device=self.device,
                dtype=torch.long,  # why long?
            )
            actions = actions.unsqueeze(0)
        return actions

    def optimize_model(self):
        # TODO add docstring
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
        reward_batch = reward_batch.unsqueeze(-1)  # TODO combine lines
        reward_batch = reward_batch.unsqueeze(-1)
        reward_batch_extended = reward_batch.expand(-1, self.N, -1).to(self.device)

        # Compute the expected Q values (All steps of each sequence are used)
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch_extended.squeeze(-1)
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
        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(), 100
        )  # TODO what is this value?
        self.optimizer.step()

    def on_train_start(self):
        # TODO add docstring
        self.train_flag = True
        self.steps_done = 0
        self.cost = []

    def on_validation_start(self):
        self.train_flag = False
        self.T_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.F_b = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.w_e = torch.empty((1, self.N), device=self.device, dtype=torch.float32)
        self.x = torch.empty((2, self.N), device=self.device, dtype=torch.float32)
        self.gear_choice_explicit = np.empty((self.N,))
        return super().on_validation_start()

    def on_episode_start(self, state: np.ndarray, env: VehicleTracking):
        self.first_time_step = True
        self.cost.append([])
        if not self.train_flag:
            self.gear = self.gear_from_velocity(state[1])
            self.gear_choice_explicit = np.ones((self.N,)) * self.gear
        return super().on_episode_start(state, env)

    def on_timestep_end(self, cost: float):
        self.cost[-1].append(cost)

    def relative_state(
        self,
        x: torch.Tensor,
        T_e: torch.Tensor,
        F_b: torch.Tensor,
        w_e: torch.Tensor,
        gear: torch.Tensor,
    ) -> torch.Tensor:
        # TODO add docstring
        d_rel = x[:, [0]] - torch.from_numpy(self.x_ref_predicition[:-1, 0]).to(
            self.device
        )
        v_rel = x[:, [1]] - torch.from_numpy(self.x_ref_predicition[:-1, 1]).to(
            self.device
        )
        v_norm = (x[:, [1]] - Vehicle.v_min) / (Vehicle.v_max - Vehicle.v_min)
        return (
            torch.cat((d_rel, v_rel, v_norm, T_e, F_b, w_e, gear), dim=1)
            .unsqueeze(0)
            .to(torch.float32)
        )  # TODO should we normalize the gears and stuff?

    def save(self, env: VehicleTracking, ep: int, path: str = ""):
        torch.save(self.policy_net.state_dict(), path + f"/policy_net_ep_{ep}.pth")
        torch.save(self.target_net.state_dict(), path + f"/target_net_ep_{ep}.pth")
        with open(path + f"/data_ep_{ep}.pkl", "wb") as f:
            pickle.dump(
                {
                    "cost": self.cost,
                    "fuel": self.fuel,
                    "T_e": self.engine_torque,
                    "w_e": self.engine_speed,
                    "x_ref": self.x_ref,
                    "R": list(env.rewards),
                    "X": list(env.observations),
                    "U": list(env.actions),
                },
                f,
            )
