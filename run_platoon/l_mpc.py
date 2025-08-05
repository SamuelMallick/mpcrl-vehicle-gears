import importlib
import os
import pickle
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
from agents.learning_agent import DistributedLearningAgent
from env import PlatoonTracking
from mpcs.fixed_gear_mpc import FixedGearMPC
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from gymnasium.wrappers import TimeLimit
from utils.parse_config import parse_config
from vehicle import Vehicle
from visualisation.plot import plot_evaluation

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

# Initialize the MPC and agent objects
mpc = SolverTimeRecorder(
    FixedGearMPC(
        N,
        optimize_fuel=True,
        convexify_fuel=False,
        solver="ipopt",
        multi_starts=config.multi_starts,
        extra_opts=config.extra_opts,
    )
)
agent = DistributedLearningAgent(
    mpc,
    np_random=np_random,
    num_vehicles=num_vehicles,
    multi_starts=config.multi_starts,
    config=config,
)

# Load policy and normalization data
state_dict = torch.load(
    f"results/{config.policy_filename}",
    weights_only=True,
    map_location="cpu",
)
with open(f"results/{config.normalization_data_filename}", "rb") as f:
    data = pickle.load(f)


# Run the evaluation
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    policy_net_state_dict=state_dict,
    normalization=data["normalization"],
    use_heuristic=True,
    heursitic_gear_priorities=["low", "mid", "high"],
)

# Collect the results
X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
fuel = [np.sum(f, axis=1) for f in fuel]
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.reference_trajectory)

solve_time = [
    np.sum(o) for o in np.split(np.array(mpc.solver_time), config.ep_len)
]  # sum vehicles solve time to get total platoon solve time

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(solve_time)}")

# Save results to pkl file
if SAVE:
    with open(f"results/platoon_l_mpc_N_{N}_c_{config.id}.pkl", "wb") as f:
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
                "heuristic": info["heuristic"] if "heuristic" in info else None,
            },
            f,
        )

# Plot results
if PLOT:
    ep = 0
    plot_evaluation(
        x_ref[ep],
        X[ep],
        U[ep],
        R[ep],
        fuel[ep],
        engine_torque[ep],
        engine_speed[ep],
    )
