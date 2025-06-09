import importlib
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())
from agents.mip_agent import MIPAgent
from env import VehicleTracking
from mpcs.mip_mpc import MIPMPC
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
from gymnasium.wrappers import TimeLimit

from vehicle import Vehicle
from visualisation.plot import plot_evaluation

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config()
else:
    from config_files.c1 import Config  # type: ignore

    config = Config()

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
            terminate_on_distance=config.terminate_on_distance,
        ),
        max_episode_steps=config.ep_len,
    )
)

mpc = SolverTimeRecorder(
    MIPMPC(
        N,
        optimize_fuel=True,
        convexify_fuel=False,
        convexify_dynamics=False,
        solver="bonmin",
        multi_starts=config.multi_starts,
        extra_opts=config.extra_opts,
    )
)
agent = MIPAgent(
    mpc,
    np_random=np_random,
    multi_starts=config.multi_starts,
    backup_mpc=None,
)
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
    allow_failure=False,
    save_every_episode=config.save_every_episode,
    log_progress=False,
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
    with open(f"minlp_mpc_N_{N}_c_{config.id}.pkl", "wb") as f:
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
