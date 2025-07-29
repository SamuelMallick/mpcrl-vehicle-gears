import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
from agents.supervised_learning_agent import SupervisedLearningAgent
from env import VehicleTracking
from mpcs.mip_mpc import MIPMPC
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from utils.parse_config import parse_config
from gymnasium.wrappers import TimeLimit

from vehicle import Vehicle

# Generate config object
config = parse_config(sys.argv)

N = config.N
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)
data_seed = 100
num_eps = 1
ep_len = 5

vehicle = Vehicle()
env: VehicleTracking = MonitorEpisodes(
    TimeLimit(
        VehicleTracking(
            vehicle,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
            windy=config.windy,
            terminate_on_distance=config.terminate_on_distance,
        ),
        max_episode_steps=ep_len,
    )
)

mpc = SolverTimeRecorder(
    MIPMPC(
        N,
        solver="gurobi",
        optimize_fuel=True,
        convexify_fuel=True,
        convexify_dynamics=True,
        multi_starts=config.multi_starts,
    )
)
agent = SupervisedLearningAgent(
    mpc, np_random=np_random, multi_starts=config.multi_starts
)

nn_inputs, nn_targets_shift, nn_targets_explicit = (
    agent.generate_supervised_training_data(
        env,
        episodes=num_eps,
        seed=data_seed,
    )
)

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.x_ref)

torch.save(
    {
        "inputs": nn_inputs,
        "targets_explicit": nn_targets_explicit,
        "targets_shift": nn_targets_shift,
    },
    f"supervised_data_{nn_inputs.shape[0]}_seed_{data_seed}.pth",
)
