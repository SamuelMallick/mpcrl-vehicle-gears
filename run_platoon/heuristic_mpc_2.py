import importlib
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())
from agents.heuristic_2_agent import DistributedHeuristic2Agent
from env import PlatoonTracking
from mpcs.fixed_gear_mpc import FixedGearMPC
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
num_vehicles = config.num_vehicles

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
gear_priority = ["low", "mid", "high"]
agent = DistributedHeuristic2Agent(
    mpc,
    np_random=np_random,
    num_vehicles=num_vehicles,
    multi_starts=config.multi_starts,
    gear_priority=gear_priority,
    inter_vehicle_distance=config.inter_vehicle_distance,
)
returns, info = agent.evaluate(
    env,
    episodes=num_eval_eps,
    seed=eval_seed,
)


X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)
fuel = list(env.fuel_consumption)
fuel = [np.sum(f, axis=1) for f in fuel]
engine_torque = list(env.engine_torque)
engine_speed = list(env.engine_speed)
x_ref = list(env.reference_trajectory)

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(mpc.solver_time)}")

if SAVE:
    with open(
        f"platoon_heuristic_2_{gear_priority}_mpc_N_{N}_c_{config.id}.pkl",
        "wb",
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
