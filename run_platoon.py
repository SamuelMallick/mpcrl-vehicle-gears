import importlib
import os
import sys
from agents import (
    Agent,
    MINLPAgent,
    HeuristicGearAgent,
    DQNAgent,
    SupervisedLearningAgent,
    DistributedAgent,
    DistributedHeuristicGearAgent,
)
from env import VehicleTracking, PlatoonTracking
from vehicle import Vehicle
from gymnasium.wrappers import TimeLimit
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import numpy as np
from visualisation.plot import plot_evaluation, plot_training
from mpc import HybridTrackingMpc, HybridTrackingFuelMpcFixedGear, TrackingMpc
import torch
import pickle
from typing import Literal

SAVE = False
PLOT = True

sim_type: Literal[
    "sl_train",
    "sl_data",
    "rl_mpc_train",
    "l_mpc_eval",
    "miqp_mpc",
    "minlp_mpc",
    "heuristic_mpc",
] = "miqp_mpc"

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config(sim_type)
else:
    from config_files.c1 import Config  # type: ignore

    config = Config(sim_type)

num_vehicles = 3
vehicles = [Vehicle() for _ in range(num_vehicles)]
ep_length = config.ep_len
num_eval_eps = 1
N = config.N
eval_seed = 10
env: PlatoonTracking = MonitorEpisodes(
    TimeLimit(
        PlatoonTracking(
            vehicles,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
            windy=config.windy,
            infinite_episodes=config.infinite_episodes,
        ),
        max_episode_steps=(
            ep_length if not config.infinite_episodes else ep_length * config.num_eps
        ),
    )
)

mpc = SolverTimeRecorder(TrackingMpc(N))
agent: Agent = DistributedHeuristicGearAgent(mpc, num_vehicles=num_vehicles)
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    save_every_episode=config.save_every_episode,
)

fuel = info["fuel"]
engine_torque = info["T_e"]
engine_speed = info["w_e"]
x_ref = info["x_ref"]
if "cost" in info:
    cost = info["cost"]


X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(
    f"average fuel = {sum([sum(fuel[i][j][k] for k in range(num_vehicles) for j in range(ep_length)) for i in range(len(fuel))]) / len(fuel)}"
)
# print(f"total mpc solve times = {sum(mpc.solver_time)}")

if SAVE:
    with open(f"results/{sim_type}_N_{N}_c_{config.id}.pkl", "wb") as f:
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
                "infeasible": info["infeasible"] if "infeasible" in info else None,
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
        [sum(fuel[ep][j][k] for k in range(num_vehicles)) for j in range(ep_length)],
        engine_torque[ep],
        engine_speed[ep],
    )
