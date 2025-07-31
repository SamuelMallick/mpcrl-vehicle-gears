import sys, os
import numpy as np

sys.path.append(os.getcwd())
from agents.agent import SingleVehicleAgent, PlatoonAgent
from mpcs.mip_mpc import MIPMPC


class MIPAgent(SingleVehicleAgent):
    """An agent that uses a mixed-integer programming solver to solve
    the optimization problem. The solver is used to find the optimal engine torque,
    braking force, and gear for the vehicle.

    Parameters
    ----------
    mpc : VehicleMPC
        The model predictive controller used to solve the optimization problem.
    np_random : np.random.Generator`
        A random number generator for generating random numbers.
    multi_starts : int, optional
        The number of multi-starts to use for the optimization problem, by default 1.
    backup_mpc : VehicleMPC, optional
        A backup model predictive controller used when the primary MPC fails to
        solve the optimization problem, by default None."""

    def __init__(
        self,
        mpc: MIPMPC,
        np_random: np.random.Generator,
        multi_starts: int = 1,
        backup_mpc: MIPMPC | None = None,
    ):
        super().__init__(mpc, np_random=np_random, multi_starts=multi_starts)
        self.backup_mpc = backup_mpc

    def solve_mpc(self, pars, vals0) -> tuple[float, float, int, dict]:
        solver = "primary"
        sol, info = self.mpc.solve(pars, vals0)

        # error handling
        accepted_knitro_statuses = [
            "KN_RC_MIP_TERM_FEAS",
            "KN_RC_TIME_LIMIT_FEAS",
        ]
        if sol.success or sol.status in accepted_knitro_statuses:

            # Append a zero time to the backup MPC since it is not used at this timestep
            if self.backup_mpc is not None and hasattr(self.backup_mpc, "solver_time"):
                self.backup_mpc.solver_time.append(0.0)

            pass  # Accept the current solution and continue

        # Handle other failure cases
        else:
            if sol.status == "TIME_LIMIT":  # timeout for gurobi
                if sol.stats["pool_sol_nr"] == 0:
                    raise ValueError("MPC failed to find feasible solution in time")
            elif sol.status == "KN_RC_TIME_LIMIT_INFEAS":
                # Special case: use heuristic 2 to generate a warm start solution
                # for the backup MPC
                gear = self.gear_from_velocity(
                    self.prev_sol.vals["x"].full()[1, 0], gear_priority="mid"
                )
                vals0[0]["gear"] = np.zeros((6, self.mpc.prediction_horizon))
                vals0[0]["gear"][gear] = 1
                sol, _ = self.backup_mpc.solve(pars, vals0)
                if not sol or sol.status not in accepted_knitro_statuses:
                    raise ValueError(
                        "Backup MPC failed to solve after primary MPC timeout"
                    )
            elif self.backup_mpc:
                # Other failure cases: try the backup MPC
                solver = "backup"
                sol, _ = self.backup_mpc.solve(pars, vals0)
                if not sol.success:
                    raise ValueError("MPC failed to solve")
            else:
                raise ValueError("MPC failed to solve")

        T_e = sol.vals["T_e"].full()[0, 0]
        F_b = sol.vals["F_b"].full()[0, 0]
        gear = np.argmax(sol.vals["gear"].full(), 0)[0]
        return T_e, F_b, gear, {"sol": sol, "solver": solver, "cost": sol.f}

    def get_action(self, state: np.ndarray) -> tuple[float, float, int, dict]:
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
        T_e, F_b, gear, info = self.solve_mpc(pars, vals0)
        self.prev_sol = self.shift_sol(info["sol"])
        return T_e, F_b, gear, info


class DistributedMIPAgent(PlatoonAgent, MIPAgent):

    def __init__(
        self,
        mpc,
        num_vehicles,
        np_random,
        inter_vehicle_distance: float,
        multi_starts=1,
        backup_mpc: MIPMPC | None = None,
    ):
        super().__init__(
            mpc=mpc,
            num_vehicles=num_vehicles,
            np_random=np_random,
            multi_starts=multi_starts,
            inter_vehicle_distance=inter_vehicle_distance,
        )
        self.backup_mpc = backup_mpc

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
            T_e, F_b, gear, info = self.solve_mpc(pars, vals0)
            T_e_list.append(T_e)
            F_b_list.append(F_b)
            gear_list.append(gear)
            self.prev_sols[i] = info["sol"]
        for i in range(self.num_vehicles):
            self.prev_sols[i] = self.shift_sol(self.prev_sols[i])
        return np.asarray(T_e_list), np.asarray(F_b_list), gear_list, {}
