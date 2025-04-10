from typing import Literal
from csnlp import Solution
import numpy as np
import torch
from agents.agent import PlatoonAgent, SingleVehicleAgent
from config_files.base import ConfigDefault
from env import VehicleTracking
from mpcs.fixed_gear_mpc import FixedGearMPC
from network import DRQN
from utils.running_mean_std import RunningMeanStd
from vehicle import Vehicle
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


class LearningAgent(SingleVehicleAgent):
    """An agent that uses a learning-based policy to select the gear-shift
    schedule. An NLP-based MPC controller then solves for the continuous
    variables, given the gear-shift schedule.

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

    def __init__(
        self,
        mpc: FixedGearMPC,
        np_random: np.random.Generator,
        config: ConfigDefault,
        multi_starts: int = 1,
    ):
        super().__init__(mpc, np_random, multi_starts)
        self._config = config

        self.prev_gear_choice_explicit = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print("Using GPU")

        self.use_heuristic = False  # changed on call to train or evaluate
        self.heuristic_flags: list[list[bool]] = (
            []
        )  # reset on call to train or evaluate
        self.infeasible_flags: list[list[bool]] = (
            []
        )  # reset on call to train or evaluate

        self.n_actions = config.n_actions

        # training hyperparameters
        self.batch_size = config.batch_size
        self.normalize = config.normalize

        # no exploration for by default
        self.eps_start = 0
        self.decay_rate = 0
        self.steps_done = 0  # gets reset on calls to train or evaluate

        # seeded initialization of networks
        seed = np_random.integers(0, 2**32 - 1)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        self.policy_net = DRQN(
            config.n_states,
            config.n_hidden,
            config.n_actions,
            config.n_layers,
            bidirectional=config.bidirectional,
        ).to(self.device)

    def evaluate(
        self,
        env: VehicleTracking,
        episodes,
        policy_net_state_dict: dict,
        use_heuristic: bool,
        heursitic_gear_priorities: list[Literal["low", "high", "mid"]],
        seed=0,
        normalization: tuple = (),
        allow_failure: bool = False,
        save_every_episode: bool = False,
        log_progress: bool = False,
    ):
        """Evaluate the agent on the vehicle tracking environment for a number of episodes,
        using the a trained policy that is provided.

        Parameters
        ----------
        env : VehicleTracking
            The environment to evaluate the agent on
        episodes : int
            The number of episodes to evaluate the agent for.
        policy_net_state_dict : dict
            The state dictionary of the policy network to load.
        use_heuristic : bool
            If True, a heuristic MPC will also solve the problem at each timestep,
            and the best solution between the heuristic and the neural network will be used.
        heursitic_gear_priorities : list[Literal["low", "high", "mid"]]
            If use_heuristic is True: For each entry in the list an MPC will also be solved
            using a fixed gear schedule determined by the heuristic (see also heuristic_2_agent.py).
        seed : int, optional
            The seed to use for the random number generator, by default 0.
        normalization : tuple, optional
            The normalization to be used on the inputs to the policy network.
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
        self.eps_start = 0  # no exploration for evaluation
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

        self.use_heuristic = use_heuristic
        self.heursitic_gear_priorities = heursitic_gear_priorities
        returns, info = super().evaluate(
            env,
            episodes,
            seed,
            allow_failure=allow_failure,
            save_every_episode=save_every_episode,
            log_progress=log_progress,
        )
        return returns, {**info, "heuristic": self.heuristic_flags}

    def solve_mpc(
        self, network_pars: dict, heuristic_pars: list[dict], vals0: dict, first_step
    ) -> tuple[float, float, int, dict]:
        heuristic_flag = False
        infeasible_flag = False
        if not first_step:
            sol = self.mpc.solve(network_pars, vals0)
            if not sol.success:
                infeasible_flag = True
        if self.use_heuristic or first_step or (not first_step and not sol.success):
            heuristic_sols = [self.mpc.solve(p, vals0) for p in heuristic_pars]
            if first_step:
                improved_sols = [True] * len(heuristic_sols)
            else:
                improved_sols = [s.f < sol.f for s in heuristic_sols]
        if (
            first_step
            or (not first_step and not sol.success)
            or (self.use_heuristic and sol.success and any(improved_sols))
        ):
            heuristic_flag = True
            indx = np.argmin([s.f for s in heuristic_sols])
            sol = heuristic_sols[indx]
            gear_choice_explicit = np.argmax(heuristic_pars[indx]["gear"], axis=0)
        else:
            gear_choice_explicit = np.argmax(network_pars["gear"], axis=0)

        if not sol.success:
            raise ValueError("MPC solver failed.")

        gear = int(gear_choice_explicit[0])
        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        return (
            T_e,
            F_b,
            gear,
            {
                "sol": sol,
                "heuristic": heuristic_flag,
                "infeasible": infeasible_flag,
                "gear_choice_explicit": gear_choice_explicit,
            },
        )

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        """Get the MPC action for the given state. Gears are chosen by
        the neural network (and also the heuristic), while the engine
        torque and brake force are then chosen by the MPC controller.

        Parameters
        ----------
        state : np.ndarray
            The current state of the vehicle.

        Returns
        -------
        tuple[float, float, int, dict]
            The engine torque, brake force, and gear chosen by the agent, and an
            info dict containing network state and action."""
        nn_state = None
        network_action = None

        vals0 = (
            [self.prev_sol.vals]
            + self.initial_guesses_vals(state, self.multi_starts - 1)
            if self.prev_sol
            else self.initial_guesses_vals(state, self.multi_starts)
        )

        if self.prev_sol:  # get gears from network for non-first time steps
            nn_state = self.relative_state(
                *self.get_nn_inputs_from_sol(
                    self.prev_sol, state, self.prev_gear_choice_explicit
                ),
                self.x_ref_predicition[:-1, :].reshape(-1, 2),
            )
            network_gear_choice_binary, network_action = (
                self.get_binary_gear_choice_from_network(nn_state, self.gear)
            )

            network_pars = {
                "x_0": state,
                "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                "T_e_prev": self.T_e_prev,
                "gear": network_gear_choice_binary,
                "gear_prev": self.gear_prev,
            }
        else:
            network_pars = {}

        heuristic_pars = []
        if not self.use_heuristic:
            heuristic_gear = self.gear_from_velocity(
                state[1].item(), "low"
            )  # low heuristic used as backup sol
            heurisitic_gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
            heurisitic_gear_choice_binary[heuristic_gear] = 1
            heuristic_pars.append(
                {
                    "x_0": state,
                    "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    "T_e_prev": self.T_e_prev,
                    "gear": heurisitic_gear_choice_binary,
                    "gear_prev": self.gear_prev,
                }
            )
        else:
            gears = []
            for gear_priority in self.heursitic_gear_priorities:
                heuristic_gear = self.gear_from_velocity(state[1].item(), gear_priority)
                if heuristic_gear not in gears:
                    gears.append(heuristic_gear)
                    heurisitic_gear_choice_binary = np.zeros(
                        (6, self.mpc.prediction_horizon)
                    )
                    heurisitic_gear_choice_binary[heuristic_gear] = 1
                    heuristic_pars.append(
                        {
                            "x_0": state,
                            "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                            "T_e_prev": self.T_e_prev,
                            "gear": heurisitic_gear_choice_binary,
                            "gear_prev": self.gear_prev,
                        }
                    )

        T_e, F_b, gear, info = self.solve_mpc(
            network_pars,
            heuristic_pars,
            vals0,
            first_step=self.prev_sol is None,
        )

        self.prev_gear_choice_explicit = np.concatenate(
            (info["gear_choice_explicit"][1:], [info["gear_choice_explicit"][-1]])
        )
        self.prev_sol = self.shift_sol(info["sol"])
        return (
            T_e,
            F_b,
            gear,
            {
                "nn_state": nn_state,
                "network_action": network_action,
                "heuristic": info["heuristic"],
                "infeasible": info["infeasible"],
            },
        )

    def get_nn_inputs_from_sol(
        self, sol: Solution, state: np.ndarray, gear_choice_explicit: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # ASSUMES SOLUTION HAS ALREADY BEEN SHIFTED (for both sol and gear_choice_explicit)
        x = sol.vals["x"].full().T
        T_e = sol.vals["T_e"].full().T
        F_b = sol.vals["F_b"].full().T
        w_e = sol.vals["w_e"].full().T
        x = torch.tensor(
            np.concatenate((state.T, x[1:-1])), dtype=torch.float32, device=self.device
        )
        T_e = torch.from_numpy(T_e).to(dtype=torch.float32, device=self.device)
        F_b = torch.from_numpy(F_b).to(dtype=torch.float32, device=self.device)
        w_e = torch.from_numpy(w_e).to(dtype=torch.float32, device=self.device)
        gear = torch.from_numpy(gear_choice_explicit).unsqueeze(1).to(self.device)
        return x, T_e, F_b, w_e, gear

    def relative_state(
        self,
        x: torch.Tensor,
        T_e: torch.Tensor,
        F_b: torch.Tensor,
        w_e: torch.Tensor,
        gear: torch.Tensor,
        x_ref: np.ndarray,
    ) -> torch.Tensor:
        d_rel = x[:, [0]] - torch.from_numpy(x_ref[:, [0]]).to(self.device)
        v_rel = x[:, [1]] - torch.from_numpy(x_ref[:, [1]]).to(self.device)
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

    def get_binary_gear_choice_from_network(
        self, nn_state: torch.Tensor, gear: int
    ) -> tuple[np.ndarray, torch.Tensor]:
        network_action = self.network_action(nn_state)
        if self.n_actions == 3:
            gear_shift = (network_action - 1).cpu().numpy()
            self.gear_choice_explicit = np.array(
                [
                    gear + np.sum(gear_shift[:, : i + 1])
                    for i in range(self.mpc.prediction_horizon)
                ]
            )
            self.gear_choice_explicit = np.clip(self.gear_choice_explicit, 0, 5)
        elif self.n_actions == 6:
            self.gear_choice_explicit = network_action.cpu().numpy().squeeze()
        return self.binary_from_explicit(self.gear_choice_explicit), network_action

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
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.argmax(2)

        actions = torch.tensor(
            [self.np_random.integers(0, self.n_actions) for _ in range(state.shape[1])],
            device=self.device,
            dtype=torch.long,  # long -> integers
        )
        return actions.unsqueeze(0)

    def binary_from_explicit(self, explicit: np.ndarray) -> np.ndarray:
        binary = np.zeros((self.n_gears, self.mpc.prediction_horizon))
        binary[explicit.astype(int), np.arange(self.mpc.prediction_horizon)] = 1
        return binary

    def on_env_step(self, env, episode, timestep, info):
        self.steps_done += 1
        if self.normalize:
            diff = info["x"] - info["x_ref"]
            self.running_mean_std.update(
                diff.T
            )  # transpose needed as the mean is taken over axis 0
        if "heuristic" in info:
            self.heuristic_flags[-1].append(info["heuristic"])
        if "infeasible" in info:
            self.infeasible_flags[-1].append(info["infeasible"])
        return super().on_env_step(env, episode, timestep, info)

    def on_episode_start(self, state, env):
        self.heuristic_flags.append([])
        self.infeasible_flags.append([])
        return super().on_episode_start(state, env)

    def on_validation_start(self):
        self.heuristic_flags: list[list[bool]] = []
        self.infeasible_flags: list[list[bool]] = []
        self.steps_done = 0

    def train_supervised(
        self,
        nn_inputs: torch.Tensor,
        nn_targets: torch.Tensor,
        train_epochs: int = 100,
        save_freq: int = 100,
        save_path: str = "",
        nn_inputs_eval: torch.Tensor = None,
        nn_targets_eval: torch.Tensor = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train the gear selection policy network using supervised learning.

        Parameters
        ----------
        nn_inputs : torch.Tensor
            The input data for the neural network, shape (num_samples, num_features).
        nn_targets : torch.Tensor
            The target data for the neural network, shape (num_samples, num_classes).
        train_epochs : int, optional
            The number of training epochs, by default 100.
        save_freq : int, optional
            The epoch frequency to save the model, by default every 100.
        save_path : str, optional
            The location at which to save the model, by default an empty string,
            resulting in saving in the cwd.
        nn_inputs_eval : torch.Tensor, optional
            Input data for evaluation during training, by default None, in which case
            no evaluation is performed.
        nn_targets_eval : torch.Tensor, optional
            Target data for evaluation during training, by default None."""
        num_steps = nn_inputs.shape[0]

        self.policy_net.to(self.device)
        nn_targets = torch.argmax(nn_targets, 2)
        s_train_tensor = nn_inputs.to(self.device, dtype=torch.float32)
        a_train_tensor = nn_targets.to(self.device, dtype=torch.long)
        dataset = TensorDataset(s_train_tensor, a_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        loss_history_train = np.empty(train_epochs, dtype=float)
        loss_history_eval = np.empty(train_epochs, dtype=float)
        criterion = nn.CrossEntropyLoss()

        if nn_inputs_eval is not None and nn_targets_eval is not None:
            nn_targets_eval = torch.argmax(nn_targets_eval, 2)
            s_eval_tensor = nn_inputs_eval.to(self.device, dtype=torch.float32)
            a_eval_tensor = nn_targets_eval.to(self.device, dtype=torch.long)

        for epoch in range(train_epochs):
            if epoch % save_freq == 0:
                torch.save(
                    self.policy_net.state_dict(),
                    f"{save_path}policy_net_{num_steps}_epoch_{epoch}.pth",
                )
            running_loss = 0.0
            self.policy_net.train()
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
            loss_history_train[epoch] = running_loss
            print(f"Epoch [{epoch+1}/{train_epochs}], Loss: {running_loss}")

            if nn_inputs_eval is not None and nn_targets_eval is not None:
                self.policy_net.eval()
                with torch.no_grad():
                    outputs = self.policy_net(s_eval_tensor)
                    loss = criterion(outputs.transpose(1, 2), a_eval_tensor)
                    loss_history_eval[epoch] = loss.item()
                    print(f"Eval loss: {loss_history_eval[epoch]}")

        torch.save(
            self.policy_net.state_dict(),
            f"{save_path}policy_net_{num_steps}_epoch_{train_epochs}.pth",
        )
        return loss_history_train, loss_history_eval


class DistributedLearningAgent(PlatoonAgent, LearningAgent):

    def __init__(
        self,
        mpc: FixedGearMPC,
        np_random: np.random.Generator,
        num_vehicles: int,
        config: ConfigDefault,
        multi_starts: int = 1,
    ):
        LearningAgent.__init__(
            self,
            mpc,
            np_random,
            config,
            multi_starts=multi_starts,
        )
        # these done manually as we don't call PlatoonAgent.__init__()
        self.d = config.inter_vehicle_distance
        self.d_arr = np.array([[self.d], [0]])
        self.num_vehicles = num_vehicles
        self.T_e_prev = [Vehicle.T_e_idle for _ in range(num_vehicles)]
        self.gear_prev = [np.zeros((6, 1)) for _ in range(num_vehicles)]
        self.prev_sols = [None for _ in range(num_vehicles)]
        self.prev_gear_choice_explicit = [None for _ in range(num_vehicles)]

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        xs = np.split(state, self.num_vehicles, axis=1)
        T_e_list = []
        F_b_list = []
        gear_list = []
        info_list = []
        for i, x in enumerate(xs):
            nn_state = None
            network_action = None

            vals0 = (
                [self.prev_sols[i].vals]
                + self.initial_guesses_vals(x, self.multi_starts - 1)
                if self.prev_sols[i]
                else self.initial_guesses_vals(x, self.multi_starts)
            )
            pars = self.get_pars(x, i)

            if self.prev_sols[i]:  # get gears from network for non-first time steps
                nn_state = self.relative_state(
                    *self.get_nn_inputs_from_sol(
                        self.prev_sols[i], x, self.prev_gear_choice_explicit[i]
                    ),
                    pars["x_ref"][:, :-1].T,
                )
                network_gear_choice_binary, network_action = (
                    self.get_binary_gear_choice_from_network(nn_state, self.gears[i])
                )

                network_pars = {**pars, "gear": network_gear_choice_binary}
            else:
                network_pars = {}

            heuristic_pars = []
            if not self.use_heuristic:
                heuristic_gear = self.gear_from_velocity(
                    x[1].item(), "low"
                )  # low heuristic used as backup sol
                heurisitic_gear_choice_binary = np.zeros(
                    (6, self.mpc.prediction_horizon)
                )
                heurisitic_gear_choice_binary[heuristic_gear] = 1
                heuristic_pars.append({**pars, "gear": heurisitic_gear_choice_binary})
            else:
                gears = []
                for gear_priority in self.heursitic_gear_priorities:
                    heuristic_gear = self.gear_from_velocity(x[1].item(), gear_priority)
                    if heuristic_gear not in gears:
                        gears.append(heuristic_gear)
                        heurisitic_gear_choice_binary = np.zeros(
                            (6, self.mpc.prediction_horizon)
                        )
                        heurisitic_gear_choice_binary[heuristic_gear] = 1
                        heuristic_pars.append(
                            {**pars, "gear": heurisitic_gear_choice_binary}
                        )

            T_e, F_b, gear, info = self.solve_mpc(
                network_pars,
                heuristic_pars,
                vals0,
                first_step=self.prev_sols[i] is None,
            )

            self.prev_gear_choice_explicit[i] = np.concatenate(
                (info["gear_choice_explicit"][1:], [info["gear_choice_explicit"][-1]])
            )
            T_e_list.append(T_e)
            F_b_list.append(F_b)
            gear_list.append(gear)
            self.prev_sols[i] = info["sol"]
            info_list.append(info)
        for i in range(self.num_vehicles):
            self.prev_sols[i] = self.shift_sol(self.prev_sols[i])
        info = {}
        for key in info_list[0].keys():
            info[key] = [info_list[i][key] for i in range(self.num_vehicles)]
        return np.asarray(T_e_list), np.asarray(F_b_list), gear_list, info

    def on_env_step(self, env, episode, timestep, info):
        # doing the below manually as we don't call LearningAgent.on_env_step()
        self.steps_done += 1
        if self.normalize:
            for i in range(self.num_vehicles):
                if i == 0:
                    diff = info["x"][i] - info["x_ref"]
                else:
                    diff = info["x"][i] - info["x"][i - 1] + self.d_arr
                self.running_mean_std.update(
                    diff.T
                )  # transpose needed as the mean is taken over axis 0
        if self.heursitic_gear_priorities:
            if "heuristic" in info:
                self.heuristic_flags[-1].append(info["heuristic"])
        return PlatoonAgent.on_env_step(self, env, episode, timestep, info)

    def on_episode_start(self, state, env):
        self.heuristic_flags.append(
            []
        )  # doing manually as we don't call LearningAgent.on_episode_start()
        return super().on_episode_start(state, env)
