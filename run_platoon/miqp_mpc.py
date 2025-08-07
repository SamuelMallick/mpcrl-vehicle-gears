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

# Simulation parameters
SAVE = config.SAVE
PLOT = config.PLOT
N = config.N
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)
eval_seed = config.eval_seed
num_eval_eps = 1
num_vehicles = config.num_vehicles

# Create the vehicles and environment
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

# Initialize the MPC and agent objects
mpc = SolverTimeRecorder(
    MIPMPC(
        N,
        optimize_fuel=True,
        convexify_fuel=True,
        convexify_dynamics=True,
        solver="gurobi",
        multi_starts=config.multi_starts,
        extra_opts=config.extra_opts,
    )
)
agent = DistributedMIPAgent(
    mpc,
    np_random=np_random,
    num_vehicles=num_vehicles,
    multi_starts=config.multi_starts,
    backup_mpc=None,
    inter_vehicle_distance=config.inter_vehicle_distance,
)

# Run the evaluation
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    allow_failure=False,
    save_every_episode=config.save_every_episode,
    log_progress=False,
)

# Collect the results
X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.reference_trajectory)

solve_time = [np.sum(o) for o in np.split(np.array(mpc.solver_time), config.ep_len)]

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(solve_time)}")

# Save results to pkl file
if config.extra_opts["gurobi"]["TimeLimit"] == 1:
    results_name = f"results/platoon_miqp_N_{N}_c_{config.id}_maxtime_1.pkl"
else:
    results_name = f"results/platoon_miqp_N_{N}_c_{config.id}.pkl"

if SAVE:
    with open(results_name, "wb") as f:
        pickle.dump(
            {
                "x_ref": x_ref,
                "X": X,
                "U": U,
                "R": R,
                "fuel": fuel,
                "T_e": engine_torque,
                "w_e": engine_speed,
                "mpc_solve_time": solve_time,
                "valid_episodes": (
                    info["valid_episodes"] if "valid_episodes" in info else None
                ),
            },
            f,
        )
