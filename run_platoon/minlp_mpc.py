import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())
from gymnasium.wrappers import TimeLimit

from agents.mip_agent import DistributedMIPAgent
from env import PlatoonTracking
from mpcs.mip_mpc import MIPMPC
from utils.parse_config import parse_config
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from vehicle import Vehicle

# Generate config object
config = parse_config(sys.argv)

# Script parameters
SAVE = config.SAVE
PLOT = config.PLOT
N = config.N
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)
eval_seed = config.eval_seed
num_eval_eps = 1
num_vehicles = config.num_vehicles

# Create vehicles and environment
vehicles = [Vehicle() for _ in range(num_vehicles)]
env: PlatoonTracking = MonitorEpisodes(
    TimeLimit(
        PlatoonTracking(
            vehicles,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
            windy=config.windy,
            terminate_on_distance=config.terminate_on_distance,
            inter_vehicle_distance=config.inter_vehicle_distance,
        ),
        max_episode_steps=config.ep_len,
    )
)

# Initialize the MIPMPC
mpc = SolverTimeRecorder(
    MIPMPC(
        N,
        optimize_fuel=True,
        convexify_fuel=False,
        convexify_dynamics=False,
        solver="knitro",
        multi_starts=config.multi_starts,
        extra_opts=config.extra_opts,
    )
)

# Add backup MPC with specific options
extra_opts_backup = {
    "knitro": {
        "mip_terminate": config.backup_minlp_mip_terminate,
        "maxtime": config.backup_minlp_maxtime,
    }
}
backup_mpc = SolverTimeRecorder(
    MIPMPC(
        N,
        optimize_fuel=True,
        convexify_fuel=False,
        convexify_dynamics=False,
        solver="knitro",
        multi_starts=config.multi_starts,
        extra_opts=extra_opts_backup,
    )
)

# Initialize the MIPAgent with the MPC and backup MPC
agent = DistributedMIPAgent(
    mpc,
    np_random=np_random,
    num_vehicles=num_vehicles,
    multi_starts=config.multi_starts,
    backup_mpc=backup_mpc,
    inter_vehicle_distance=config.inter_vehicle_distance,
)

# Run the agent evaluation
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    allow_failure=False,
    save_every_episode=config.save_every_episode,
    log_progress=False,
    print_actions=False,
)

# Extract relevant data from the env object
X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
fuel = [np.sum(f, axis=1) for f in fuel]
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.reference_trajectory)

# Compute MPC total solve time (including backup MPC if used)
t_primary_mpc = np.array(mpc.solver_time)
if backup_mpc is not None and hasattr(backup_mpc, "solver_time"):
    t_backup_mpc = np.array(backup_mpc.solver_time)
else:
    t_backup_mpc = np.zeros(len(t_primary_mpc))
t_total_mpc = t_primary_mpc + t_backup_mpc

solve_time = [
    np.sum(o) for o in np.split(t_total_mpc, config.ep_len)  # split in N=ep_len chunks
]  # sum the solve time of each vehicle in platoon to get total platoon solve time

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(solve_time)}")

# Save results to pickle file
# NOTE: times are saved vehicle-wise and not platoon-wise, i.e., t_primary_mpc has
# length num_eval_eps * ep_len * num_vehicles.
if SAVE:
    with open(f"results/platoon_minlp_N_{N}_c_{config.id}.pkl", "wb") as f:
        pickle.dump(
            {
                "x_ref": x_ref,
                "X": X,
                "U": U,
                "R": R,
                "fuel": fuel,
                "T_e": engine_torque,
                "w_e": engine_speed,
                "mpc_solve_time": t_total_mpc,
                "t_primary_mpc": t_primary_mpc,
                "t_backup_mpc": t_backup_mpc,
                "valid_episodes": (
                    info["valid_episodes"] if "valid_episodes" in info else None
                ),
            },
            f,
        )
