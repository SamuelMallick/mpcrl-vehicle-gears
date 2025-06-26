import importlib
import os
import pickle
import sys
import warnings

import numpy as np
import torch

sys.path.append(os.getcwd())
from gymnasium.wrappers import TimeLimit

from agents.learning_agent import LearningAgent
from env import VehicleTracking
from mpcs.fixed_gear_mpc import FixedGearMPC
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from vehicle import Vehicle
from visualisation.plot import plot_evaluation
from config_files.evaluations.c1 import Config  # type: ignore

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config()
else:
    warnings.warn("No config file passed on command line, using default config file.")
    config = Config()

# if additional arguments are passed, assign them
config_number = 0
config_seed = 0
config_step = 0
if len(sys.argv) > 2:
    try:
        config_number = int(sys.argv[2])
        config_seed = int(sys.argv[3])
        config_step = int(sys.argv[4])
    except IndexError:
        raise ValueError(
            "Not enough arguments provided. Check the bash script run_eval_single.sh."
            "Expected: <config_number> <config_seed> <config_step>"
        )

# Default parameters values (c2_seed4_step4000000)
if config_number == 0:
    config_number = 2
if config_seed == 0:
    config_seed = 4
if config_step == 0:
    config_step = 4000000

# Compose filenames for policy, normalization data, and results folder
policy_filename = (
    f"c{config_number}/c{config_number}_seed{config_seed}/"
    f"policy_net_step_{config_step}.pth"
)
normalization_data_filename = (
    f"c{config_number}/c{config_number}_seed{config_seed}/data_step_{config_step}.pkl"
)
results_folder_name = f"eval_l_mpc/eval_c{config_number}_s{config_seed}_t{config_step}"

# Override everything with manual names if needed (uncomment and set values)
# policy_filename = ""
# normalization_data_filename = ""
# results_folder_name = ""

# Create results folder if it does not exist
if not os.path.exists(f"results/{results_folder_name}"):
    os.makedirs(f"results/{results_folder_name}")

# Other configuration parameters
SAVE = config.SAVE
PLOT = config.PLOT
N = config.N
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)
eval_seed = config.eval_seed
num_eval_eps = 1

vehicle = Vehicle()
env: VehicleTracking = MonitorEpisodes(
    TimeLimit(
        VehicleTracking(
            vehicle,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
            windy=config.windy,
            terminate_on_distance=False,
        ),
        max_episode_steps=config.ep_len,
    )
)

use_heuristic = True
heursitic_gear_priorities = ["low", "mid", "high"]
mpc = SolverTimeRecorder(
    FixedGearMPC(
        N,
        solver="ipopt",
        optimize_fuel=True,
        convexify_fuel=False,
        multi_starts=(
            config.multi_starts
            if not use_heuristic
            else len(heursitic_gear_priorities) * config.multi_starts
        ),  # also multistarts for gear schedules
        extra_opts=config.extra_opts,
    )
)
agent = LearningAgent(
    mpc, np_random=np_random, config=config, multi_starts=config.multi_starts
)

state_dict = torch.load(
    f"results/{policy_filename}",
    weights_only=True,
    map_location="cpu",
)
with open(f"results/{normalization_data_filename}", "rb") as f:
    data = pickle.load(f)

returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    policy_net_state_dict=state_dict,
    normalization=data["normalization"],
    use_heuristic=True,
    heursitic_gear_priorities=heursitic_gear_priorities,
)

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.reference_trajectory)

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(mpc.solver_time)}")

if SAVE:
    with open(
        f"results/{results_folder_name}/l_mpc_N_{N}_c_{config.id}.pkl", "wb"
    ) as f:
        pickle.dump(
            {
                "x_ref": x_ref,
                "X": X,
                "U": U,
                "R": R,
                "fuel": fuel,
                "T_e": engine_torque,
                "w_e": engine_speed,
                "mpc_solve_time": mpc.solver_time,
                "valid_episodes": (
                    info["valid_episodes"] if "valid_episodes" in info else None
                ),
                "heuristic": info["heuristic"] if "heuristic" in info else None,
            },
            f,
        )

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
