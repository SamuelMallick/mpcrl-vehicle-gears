import importlib
import os
import sys
from agent import Agent, MINLPAgent, HeuristicGearAgent
from dqn_agent import DQNAgent, DRQN
from env import VehicleTracking
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

SAVE = True
PLOT = False

sim_type: Literal[
    "sl_train",
    "sl_data",
    "rl_mpc_train",
    "rl_mpc_eval",
    "miqp_mpc",
    "minlp_mpc",
    "heuristic_mpc",
] = "sl_train"

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config(sim_type)
else:
    from config_files.c1 import Config  # type: ignore

    config = Config(sim_type)

vehicle = Vehicle()
ep_length = config.ep_len
num_eval_eps = 100
N = config.N
seed = 10
env = MonitorEpisodes(
    TimeLimit(
        VehicleTracking(
            vehicle,
            episode_len=ep_length,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
        ),
        max_episode_steps=ep_length,
    )
)
np_random = np.random.default_rng(seed)

if (
    sim_type == "rl_mpc_train"
    or sim_type == "rl_mpc_eval"
    or sim_type == "sl_data"
    or sim_type == "sl_train"
):
    mpc = SolverTimeRecorder(
        HybridTrackingFuelMpcFixedGear(N, optimize_fuel=True, convexify_fuel=False)
    )
    agent = DQNAgent(mpc, np_random, config=config)
    if sim_type == "rl_mpc_train":
        os.makedirs(f"results/{config.id}", exist_ok=True)
        init_state_dict = config.init_state_dict

        num_eps = config.num_eps
        returns, info = agent.train(
            env,
            episodes=num_eps,
            ep_len=ep_length,
            exp_zero_steps=config.esp_zero_steps,
            save_freq=1000,
            save_path=f"results/{config.id}",
            seed=seed,
            init_state_dict=init_state_dict,
        )
    elif sim_type == "rl_mpc_eval":
        state_dict = torch.load(
            f"results/sl_data/explicit/policy_net_ep_300_epoch_1200.pth",
            weights_only=True,
            map_location="cpu",
        )
        # dqn = DRQN(8, 256, 6, 4, True)
        # state_dict = dqn.state_dict()
        returns, info = agent.evaluate(
            env,
            episodes=num_eval_eps,
            policy_net_state_dict=state_dict,
            seed=seed,
        )
    elif sim_type == "sl_data":
        num_data_gather_eps = 100
        agent.generate_supervised_data(
            env,
            episodes=num_data_gather_eps,
            ep_len=ep_length,
            mpc=config.expert_mpc,
            seed=seed,
            save_path=f"results/",
            save_freq=100,
        )
    else:
        nn_inputs = None
        nn_targets = None
        directory = "results/sl_data"
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                data = torch.load(filepath, map_location="cpu")
                if nn_inputs is None:
                    nn_inputs = data["inputs"]
                    if config.n_actions == 3:
                        nn_targets = data["targets_shift"]
                    else:
                        nn_targets = data["targets_explicit"]
                else:
                    nn_inputs = torch.cat((nn_inputs, data["inputs"]))
                    if config.n_actions == 3:
                        nn_targets = torch.cat((nn_targets, data["targets_shift"]))
                    else:
                        nn_targets = torch.cat((nn_targets, data["targets_explicit"]))
        running_loss, loss_history = agent.train_supervised(
            nn_inputs, nn_targets, train_epochs=5000
        )
        with open(f"results/{config.id}_loss_history_5000.pkl", "wb") as f:
            pickle.dump(loss_history, f)

elif sim_type == "miqp_mpc":
    mpc = SolverTimeRecorder(
        HybridTrackingMpc(
            N, optimize_fuel=True, convexify_fuel=True, convexify_dynamics=True
        )
    )
    agent = MINLPAgent(mpc)
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=seed)
elif sim_type == "minlp_mpc":
    mpc = SolverTimeRecorder(
        HybridTrackingMpc(
            N, optimize_fuel=True, convexify_fuel=False, convexify_dynamics=False
        )
    )
    agent = MINLPAgent(mpc)
    returns, info = agent.evaluate(
        env, episodes=num_eval_eps, seed=seed, allow_failure=True
    )
elif sim_type == "heuristic_mpc":
    mpc = SolverTimeRecorder(TrackingMpc(N))
    gear_priority = "low"
    sim_type = f"{sim_type}_{gear_priority}"
    agent = HeuristicGearAgent(mpc, gear_priority=gear_priority)
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=seed)


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
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(mpc.solver_time)}")

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
        fuel[ep],
        engine_torque[ep],
        engine_speed[ep],
    )
    # plot_training(
    #     [sum(cost[i]) for i in range(len(cost))],
    #     [sum(fuel[i]) for i in range(len(fuel))],
    #     [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    #     [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
    # )
