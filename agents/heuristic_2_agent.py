from typing import Literal
import numpy as np
from agents.agent import PlatoonAgent, SingleVehicleAgent
from mpcs.fixed_gear_mpc import FixedGearMPC


class Heuristic2Agent(SingleVehicleAgent):
    """An agent that fixes a gear-shift schedule with a constant gear,
    chosen via a heuristic, and solves a NLP for the engine torque and braking
    force.

    Parameters
    ----------
    mpc : FixedGearMPC
        The model predictive controller used to solve the optimization problem.
    np_random : np.random.Generator
        The random number generator used to generate initial guesses.
    gear_priority : list[Literal["low", "high", "mid"]]
        The priority of the gear to use. An MPC will be solved for each
        entry in the list. If "low", the agent will favour lower gears, if "high",
        the agent will favour higher gears, and if "mid", the agent will favour the
        middle gear that satisfies the engine speed limit.
    multi_starts : int, optional
        The number of initial guesses to generate, by default 1."""

    def __init__(
        self,
        mpc: FixedGearMPC,
        np_random: np.random.Generator,
        gear_priority: list[Literal["low", "mid", "high"]],
        multi_starts: int = 1,
    ):
        self.gear_priority = gear_priority
        super().__init__(mpc, np_random=np_random, multi_starts=multi_starts)

    def solve_mpc(self, pars, vals0) -> tuple[float, float, int, dict]:
        sol = self.mpc.solve(pars, vals0)
        if not sol.success:
            raise ValueError("MPC failed to solve")
        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        gear = np.argmax(pars["gear"], 0)[0]
        return T_e, F_b, gear, {"sol": sol}

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        best_sol = None
        best_T_e = None
        best_F_b = None
        best_gear = None
        gears = []
        vals0 = (
            [self.prev_sol.vals]
            + self.initial_guesses_vals(state, self.multi_starts - 1)
            if self.prev_sol
            else self.initial_guesses_vals(state, self.multi_starts)
        )
        for gear_priority in self.gear_priority:
            gear = self.gear_from_velocity(state[1].item(), gear_priority)
            if gear not in gears:
                gears.append(gear)
                gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
                gear_choice_binary[gear] = 1
                pars = {
                    "x_0": state,
                    "x_ref": self.x_ref_predicition.T.reshape(2, -1),
                    "T_e_prev": self.T_e_prev,
                    "gear": gear_choice_binary,
                    "gear_prev": self.gear_prev,
                }
                T_e, F_b, gear, info = self.solve_mpc(pars, vals0)
                if best_sol is None or info["sol"].f < best_sol.f:
                    best_sol = info["sol"]
                    best_T_e = T_e
                    best_F_b = F_b
                    best_gear = gear
        self.prev_sol = self.shift_sol(best_sol)
        return best_T_e, best_F_b, best_gear, {}


class DistributedHeuristic2Agent(PlatoonAgent, Heuristic2Agent):

    def __init__(
        self,
        mpc,
        num_vehicles,
        np_random,
        inter_vehicle_distance: float,
        gear_priority: list[Literal["low", "mid", "high"]],
        multi_starts=1,
    ):
        self.gear_priority = gear_priority
        super().__init__(
            mpc=mpc,
            num_vehicles=num_vehicles,
            np_random=np_random,
            multi_starts=multi_starts,
            inter_vehicle_distance=inter_vehicle_distance,
        )

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        xs = np.split(state, self.num_vehicles, axis=1)
        T_e_list = []
        F_b_list = []
        gear_list = []
        for i, x in enumerate(xs):
            pars = self.get_pars(x, i)
            vals0 = (
                [self.prev_sols[i].vals]
                + self.initial_guesses_vals(x, self.multi_starts - 1)
                if self.prev_sols[i]
                else self.initial_guesses_vals(x, self.multi_starts)
            )
            best_sol = None
            best_T_e = None
            best_F_b = None
            best_gear = None
            gears = []
            for gear_priority in self.gear_priority:
                gear = self.gear_from_velocity(x[1].item(), gear_priority)
                if gear not in gears:
                    gears.append(gear)
                    gear_choice_binary = np.zeros((6, self.mpc.prediction_horizon))
                    gear_choice_binary[gear] = 1
                    _pars = {**pars, "gear": gear_choice_binary}
                    T_e, F_b, gear, info = self.solve_mpc(_pars, vals0)
                    if best_sol is None or info["sol"].f < best_sol.f:
                        best_sol = info["sol"]
                        best_T_e = T_e
                        best_F_b = F_b
                        best_gear = gear
            T_e_list.append(best_T_e)
            F_b_list.append(best_F_b)
            gear_list.append(best_gear)
            self.prev_sols[i] = best_sol
        for i in range(self.num_vehicles):
            self.prev_sols[i] = self.shift_sol(self.prev_sols[i])
        return np.asarray(T_e_list), np.asarray(F_b_list), gear_list, {}
