from bisect import bisect_right
from typing import Literal
import numpy as np
from agents.agent import PlatoonAgent, SingleVehicleAgent
from mpcs.nonlinear_mpc import NonlinearMPC
from vehicle import Vehicle


class Heuristic1Agent(SingleVehicleAgent):
    """An agent that solves a continuous optimization problem with an approximate
    model, considering only continuous variables. Then it uses a heuristic to
    determine the gear to use, based on the engine speed and traction force.

    Parameters
    ----------
    mpc : NonlinearMPC
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
        mpc: NonlinearMPC,
        np_random: np.random.Generator,
        gear_priority: Literal["low", "high", "mid"] = "mid",
        multi_starts: int = 1,
    ):
        self.gear_priority = gear_priority
        super().__init__(mpc, np_random=np_random, multi_starts=multi_starts)

    def initial_guesses_vals(self, state, F_trac_max: float, num_guesses=5):
        starts = super().initial_guesses_vals(state, num_guesses)
        for i in range(num_guesses):
            starts[i]["F_trac"] = self.np_random.uniform(
                self.mpc.F_trac_min, F_trac_max, (1, self.mpc.prediction_horizon)
            )
        return starts

    def gear_from_velocity_and_traction(self, v: float, F_trac: float) -> int:
        # TODO check the buffers here
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

    def solve_mpc(self, pars, vals0, T_e_prev) -> tuple[float, float, int, dict]:
        sol = self.mpc.solve(pars, vals0)
        if not sol.success:
            raise ValueError("MPC failed to solve")
        F_trac = sol.vals["F_trac"].full()[0, 0]
        gear = self.gear_from_velocity_and_traction(pars["x_0"][1], F_trac)
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
        T_e = np.clip(T_e - T_e_prev, -Vehicle.dT_e_max, Vehicle.dT_e_max) + T_e_prev
        return T_e, F_b, gear, {"sol": sol}

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        # get highest allowed gear for the given velocity, then use that to limit the traction force
        idx = bisect_right(self.max_v_per_gear, state[1].item())
        n = Vehicle.z_f * Vehicle.z_t[idx] / Vehicle.r_r
        F_trac_max = Vehicle.T_e_max * n

        pars = {
            "x_0": state,
            "x_ref": self.x_ref_predicition.T.reshape(2, -1),
            "F_trac_max": F_trac_max,
        }
        vals0 = (
            [self.prev_sol.vals]
            + self.initial_guesses_vals(state, F_trac_max, self.multi_starts - 1)
            if self.prev_sol
            else self.initial_guesses_vals(state, F_trac_max, self.multi_starts)
        )
        T_e, F_b, gear, info = self.solve_mpc(pars, vals0, self.T_e_prev)
        self.prev_sol = self.shift_sol(info["sol"])
        return T_e, F_b, gear, {}


class DistributedHeuristic1Agent(PlatoonAgent, Heuristic1Agent):

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
        xs = np.split(state, self.num_vehicles, axis=1)
        T_e_list = []
        F_b_list = []
        gear_list = []
        for i, x in enumerate(xs):
            idx = bisect_right(self.max_v_per_gear, x[1].item())
            n = Vehicle.z_f * Vehicle.z_t[idx] / Vehicle.r_r
            F_trac_max = Vehicle.T_e_max * n
            pars = self.get_pars(x, i)
            pars["F_trac_max"] = F_trac_max
            vals0 = (
                [self.prev_sols[i].vals]
                + self.initial_guesses_vals(x, self.multi_starts - 1)
                if self.prev_sols[i]
                else self.initial_guesses_vals(x, self.multi_starts)
            )
            T_e, F_b, gear, info = self.solve_mpc(pars, vals0, self.T_e_prev[i])
            T_e_list.append(T_e)
            F_b_list.append(F_b)
            gear_list.append(gear)
            self.prev_sols[i] = info["sol"]
        for i in range(self.num_vehicles):
            self.prev_sols[i] = self.shift_sol(self.prev_sols[i])
        return np.asarray(T_e_list), np.asarray(F_b_list), gear_list, {}
