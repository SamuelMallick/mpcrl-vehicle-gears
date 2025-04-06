import torch
from agents.learning_agent import LearningAgent
from config_files.base import ConfigDefault
from env import VehicleTracking
from mpcs.mip_mpc import MIPMPC
import numpy as np


class SupervisedLearningAgent(LearningAgent):
    """An agent used to generate supervised learning data for gear selection.
    Training an agent based on supervised learning data is done with the
    LearningAgent class.

    Parameters
    ----------
    mpc : MIPMPC
        The mpc controller that optimized the gears, generating data.
    np_random : np.random.Generator
        The random number generator used for sampling.
    multi_starts : int, optional
        The number of initial guesses for the solver, by default 1."""

    def __init__(
        self,
        mpc: MIPMPC,
        np_random: np.random.Generator,
        multi_starts: int = 1,
    ):
        # default config used as it is irrelevant for data generation
        super().__init__(mpc, np_random, ConfigDefault(), multi_starts)
        self.normalize = False  # never normalize the data during data gathering

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        if self.prev_sol:
            nn_state = self.relative_state(
                *self.get_nn_inputs_from_sol(
                    self.prev_sol, state, self.prev_gear_choice_explicit
                ),
                self.x_ref_predicition[:-1, :].reshape(-1, 2),
            )
        else:
            nn_state = None
        pars = {
            "x_0": state,
            "x_ref": self.x_ref_predicition.T.reshape(2, -1),
            "T_e_prev": self.T_e_prev,
            "gear_prev": self.gear_prev,
        }
        vals0 = (
            [self.prev_sol.vals]
            + self.initial_guesses_vals(state, self.multi_starts - 1)
            if self.prev_sol
            else self.initial_guesses_vals(state, self.multi_starts)
        )
        solver = "primary"
        sol = self.mpc.solve(pars, vals0)

        # special check for knitro timeout
        if (
            not sol.success and sol.status != "KN_RC_TIME_LIMIT_FEAS"
        ) and self.backup_mpc:
            solver = "backup"
            sol = self.backup_mpc.solve(pars, vals0)
            if not sol.success:
                raise ValueError("MPC failed to solve")
        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        gear_choice_explicit = np.argmax(sol.vals["gear"].full(), axis=0)
        gear = gear_choice_explicit[0]

        self.prev_gear_choice_explicit = np.concatenate(
            (gear_choice_explicit[1:], [gear_choice_explicit[-1]])
        )
        self.prev_sol = self.shift_sol(sol)
        return (
            T_e,
            F_b,
            gear,
            {"sol": sol, "solver": solver, "cost": sol.f, "nn_state": nn_state},
        )

    def generate_supervised_training_data(
        self, env: VehicleTracking, episodes: int, seed: int = 0
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Generates supervised training data for gear selection.

        Parameters
        ----------
        env : VehicleTracking
            The environment used for generating training data.
        episodes : int
            The number of episodes to run for generating training data.
        seed : int, optional
            The seed for the random number generator, by default 0.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            A tuple containing the input data (states) and target data as gear shifts
            explicit gear choices.
        """
        self.nn_inputs = []
        self.nn_targets_shift = []
        self.nn_targets_explicit = []
        self.evaluate(env, episodes, seed=seed, policy_net_state_dict={})
        return (
            torch.stack(self.nn_inputs),
            torch.stack(self.nn_targets_shift),
            torch.stack(self.nn_targets_explicit),
        )

    def on_timestep_end(
        self,
        timestep: int,
        prev_state: np.ndarray,
        state: np.ndarray,
        action: tuple[float, float, int],
        info: dict,
    ) -> None:
        if self.nn_inputs is not None:
            sol = info["sol"]
            nn_state = info["nn_state"]
            if timestep != 0:
                optimal_gears = np.argmax(sol.vals["gear"].full(), 0)
                # shift command
                optimal_gears_extended = np.insert(optimal_gears, 0, self.gear)
                gear_shift = optimal_gears_extended[1:] - optimal_gears_extended[:-1]
                action = torch.zeros(
                    (self.mpc.prediction_horizon, 3), dtype=torch.float32
                )
                action[range(self.mpc.prediction_horizon), gear_shift + 1] = 1
                self.nn_targets_shift.append(action)
                # explicit command
                action = torch.zeros(
                    (self.mpc.prediction_horizon, 6), dtype=torch.float32
                )
                action[range(self.mpc.prediction_horizon), optimal_gears] = 1
                self.nn_targets_explicit.append(action)

                self.nn_inputs.append(nn_state)
