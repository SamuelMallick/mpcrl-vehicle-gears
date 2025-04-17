import importlib
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())
from agents.dqn_agent import DQNAgent
from env import VehicleTracking
from mpcs.fixed_gear_mpc import FixedGearMPC
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from gymnasium.wrappers import TimeLimit

from vehicle import Vehicle

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config()
else:
    from config_files.c1 import Config  # type: ignore

    config = Config()

N = config.N
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)
training_seed = config.train_seed

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
        max_episode_steps=(
            config.ep_len if config.finite_episodes else config.max_train_steps
        ),
    )
)

mpc = SolverTimeRecorder(
    FixedGearMPC(
        N,
        solver="ipopt",
        optimize_fuel=True,
        convexify_fuel=False,
        multi_starts=config.multi_starts,
        extra_opts=config.extra_opts,
    )
)
agent = DQNAgent(
    mpc, np_random=np_random, config=config, multi_starts=config.multi_starts
)
os.makedirs(f"results/{config.id}", exist_ok=True)
agent.train(
    env,
    episodes=config.max_episodes,
    seed=training_seed,
    save_freq=25000,
    save_path=f"results/{config.id}",
    exp_zero_steps=config.exp_zero_steps,
    max_learning_steps=config.max_train_steps,
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
