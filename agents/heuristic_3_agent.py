from typing import Literal
import numpy as np
from agents.agent import PlatoonAgent, SingleVehicleAgent
from mpcs.fixed_gear_mpc import FixedGearMPC


class Heuristic3Agent(SingleVehicleAgent):
    """An agent that fixes a gear-shift schedule with gears depending on the
    shifted predicted velocity from the previous time step. The gear is
    chosen via a heuristic, and an NLP is solved for the engine torque and braking
    force.

    Parameters
    ----------
    mpc : FixedGearMPC
        The model predictive controller used to solve the optimization problem.
    np_random : np.random.Generator
        The random number generator used to generate initial guesses.
    multi_starts : int, optional
        The number of initial guesses to generate, by default 1.
    gear_priority : Literal["low", "high", "mid"], optional
        The priority of the gear to use, by default "mid". If "low", the agent
        will favour lower gears, if "high", the agent will favour higher gears,
        and if "mid", the agent will favour the middle gear that satisfies the
        engine speed limit."""

    def __init__(
        self,
        mpc: FixedGearMPC,
        np_random: np.random.Generator,
        gear_priority: Literal["low", "high", "mid"] = "low",
        multi_starts: int = 1,
    ):
        self.gear_priority = gear_priority
        super().__init__(mpc, np_random=np_random, multi_starts=multi_starts)

    def solve_mpc(self, pars, vals0) -> tuple[float, float, int, dict]:
        sol = self.mpc.solve(pars, vals0)
        if not sol.success:
            raise ValueError("MPC failed to solve")
        if not sol.success:
            raise ValueError("MPC failed to solve")
        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        gear = np.argmax(pars["gear"], 0)[0]
        return T_e, F_b, gear, {"sol": sol}

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        if not self.prev_sol:
            gear = self.gear_from_velocity(state[1].item(), self.gear_priority)
            gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
            gear_choice_binary[gear] = 1
        else:
            gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
            for i in range(self.mpc.prediction_horizon):
                gear = self.gear_from_velocity(self.prev_sol.vals["x"][1, i])
                gear_choice_binary[gear, i] = 1
        pars = {
            "x_0": state,
            "x_ref": self.x_ref_predicition.T.reshape(2, -1),
            "T_e_prev": self.T_e_prev,
            "gear": gear_choice_binary,
            "gear_prev": self.gear_prev,
        }
        vals0 = (
            [self.prev_sol.vals]
            + self.initial_guesses_vals(state, self.multi_starts - 1)
            if self.prev_sol
            else self.initial_guesses_vals(state, self.multi_starts)
        )
        T_e, F_b, gear, info = self.solve_mpc(pars, vals0)
        self.prev_sol = self.shift_sol(info["sol"])
        return T_e, F_b, gear, {}


class DistributedHeuristic3Agent(PlatoonAgent, Heuristic3Agent):

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        xs = np.split(state, self.num_vehicles, axis=1)
        T_e_list = []
        F_b_list = []
        gear_list = []
        for i, x in enumerate(xs):
            if not self.prev_sols[i]:
                gear = self.gear_from_velocity(x[1].item(), self.gear_priority)
                gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
                gear_choice_binary[gear] = 1
            else:
                gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
                for i in range(self.mpc.prediction_horizon):
                    gear = self.gear_from_velocity(self.prev_sols[i].vals["x"][1, i])
                    gear_choice_binary[gear, i] = 1
            pars = self.get_pars(x, i)
            pars["gear"] = gear_choice_binary
            vals0 = (
                [self.prev_sols[i].vals]
                + self.initial_guesses_vals(x, self.multi_starts - 1)
                if self.prev_sols[i]
                else self.initial_guesses_vals(x, self.multi_starts)
            )
            T_e, F_b, gear, info = self.solve_mpc(pars, vals0)
            T_e_list.append(T_e)
            F_b_list.append(F_b)
            gear_list.append(gear)
            self.prev_sols[i] = info["sol"]
        for i in range(self.num_vehicles):
            self.prev_sols[i] = self.shift_sol(self.prev_sols[i])
        return np.asarray(T_e_list), np.asarray(F_b_list), gear_list, {}
