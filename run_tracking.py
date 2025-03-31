import importlib
import os
import sys
from agents import (
    Agent,
    MINLPAgent,
    HeuristicGearAgent,
    HeuristicGearAgent2,
    DQNAgent,
    SupervisedLearningAgent,
)
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
    "l_mpc_eval",
    "miqp_mpc",
    "minlp_mpc",
    "heuristic_mpc",
    "heuristic_mpc_2",
] = "miqp_mpc"

EVAL = True

# if a config file passed on command line, otherwise use default config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"config_files.{config_file}")
    config = mod.Config(sim_type)
else:
    from config_files.c25 import Config  # type: ignore

    config = Config(sim_type)

vehicle = Vehicle()
ep_length = config.ep_len
num_eval_eps = 1 if config.infinite_episodes else 100
N = config.N
eval_seed = 10
env: VehicleTracking = MonitorEpisodes(
    TimeLimit(
        VehicleTracking(
            vehicle,
            prediction_horizon=N,
            trajectory_type=config.trajectory_type,
            windy=config.windy,
            infinite_episodes=(
                False if sim_type == "l_mpc_eval" else config.infinite_episodes
            ),
        ),
        max_episode_steps=(
            ep_length if not config.infinite_episodes or EVAL else config.max_steps
        ),
    )
)
agent: Agent | None = None
seed = 0  # seed 0 used for generator
np_random = np.random.default_rng(seed)

if sim_type == "rl_mpc_train" or sim_type == "l_mpc_eval":
    mpc = SolverTimeRecorder(
        HybridTrackingFuelMpcFixedGear(
            N,
            optimize_fuel=True,
            convexify_fuel=False,
            multi_starts=config.multi_starts,
        ),
    )
    agent = DQNAgent(mpc, np_random, config=config, multi_starts=config.multi_starts)
    if sim_type == "rl_mpc_train":
        os.makedirs(f"results/{config.id}", exist_ok=True)
        init_state_dict = config.init_state_dict

        num_eps = config.num_eps
        returns, info = agent.train(
            env,
            episodes=num_eps,
            exp_zero_steps=config.esp_zero_steps,
            save_freq=50000,
            save_path=f"results/{config.id}",
            seed=0,  # seed 0 used for training
            init_state_dict=init_state_dict,
            max_learning_steps=config.max_steps,
        )
    elif sim_type == "l_mpc_eval":
        state_dict = torch.load(
            f"dev/results/25/policy_net_step_4050000.pth",
            weights_only=True,
            map_location="cpu",
        )
        # state_dict = torch.load(
        #     f"results/11/policy_net_ep_37000.pth",
        #     weights_only=True,
        #     map_location="cpu",
        # )
        with open(f"dev/results/25/data_step_4050000.pkl", "rb") as f:
            data = pickle.load(f)
        # dqn = DRQN(8, 256, 6, 4, True)
        # state_dict = dqn.state_dict()
        returns, info = agent.evaluate(
            env,
            episodes=num_eval_eps,
            policy_net_state_dict=state_dict,
            seed=eval_seed,  # seed 10 used for evaluation
            normalization=data["normalization"],
        )
if sim_type == "sl_train" or sim_type == "sl_data":
    mpc = SolverTimeRecorder(
        HybridTrackingFuelMpcFixedGear(N, optimize_fuel=True, convexify_fuel=False)
    )
    seed = 0  # seed 0 used for agents
    np_random = np.random.default_rng(seed)
    config.normalize = False  # normalization not used for supervised learning
    agent = SupervisedLearningAgent(mpc, np_random, config=config)
    if sim_type == "sl_data":
        num_data_gather_eps = 1
        seed = 110  # different seed used for data generation
        agent.generate_supervised_data(
            env,
            episodes=num_data_gather_eps,
            ep_len=300 * 100,
            mpc=config.expert_mpc,
            seed=seed,
            save_path=f"results/",
            save_freq=10000,
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
        train_epochs = 5000
        running_loss, loss_history = agent.train(
            nn_inputs, nn_targets, train_epochs=train_epochs, save_path=f"{config.id}_"
        )
        with open(f"results/{config.id}_loss_history_{train_epochs}.pkl", "wb") as f:
            pickle.dump(loss_history, f)

elif sim_type == "miqp_mpc":
    mpc = SolverTimeRecorder(
        HybridTrackingMpc(
            N,
            optimize_fuel=True,
            convexify_fuel=True,
            convexify_dynamics=True,
            multi_starts=config.multi_starts,
        )
    )
    agent = MINLPAgent(mpc, multi_starts=config.multi_starts, np_random=np_random)
    returns, info = agent.evaluate(
        env,
        episodes=num_eval_eps,
        seed=eval_seed,
        save_every_episode=config.save_every_episode,
    )
elif sim_type == "minlp_mpc":
    mpc = SolverTimeRecorder(
        HybridTrackingMpc(
            N,
            optimize_fuel=True,
            convexify_fuel=False,
            convexify_dynamics=False,
            solver="knitro",
        )
    )
    backup_mpc = SolverTimeRecorder(
        HybridTrackingMpc(
            N,
            optimize_fuel=True,
            convexify_fuel=False,
            convexify_dynamics=False,
            solver="bonmin",
        )
    )
    agent = MINLPAgent(mpc, backup_mpc=backup_mpc)
    returns, info = agent.evaluate(
        env,
        episodes=num_eval_eps,
        seed=eval_seed,
        allow_failure=True,
        save_every_episode=config.save_every_episode,
        log_progress=True,
    )
elif sim_type == "heuristic_mpc":
    mpc = SolverTimeRecorder(TrackingMpc(N, multi_starts=config.multi_starts))
    gear_priority = "low"
    sim_type = f"{sim_type}_{gear_priority}"
    agent = HeuristicGearAgent(
        mpc,
        multi_starts=config.multi_starts,
        gear_priority=gear_priority,
        np_random=np_random,
    )
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=eval_seed)

elif sim_type == "heuristic_mpc_2":
    mpc = SolverTimeRecorder(
        HybridTrackingFuelMpcFixedGear(
            N,
            optimize_fuel=True,
            convexify_fuel=False,
            multi_starts=config.multi_starts,
        )
    )
    gear_priority = "low"
    sim_type = f"{sim_type}_{gear_priority}"
    agent = HeuristicGearAgent2(
        mpc,
        gear_priority=gear_priority,
        np_random=np_random,
        multi_starts=config.multi_starts,
    )
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=eval_seed)


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
    with open(f"{sim_type}_N_{N}_c_{config.id}_s_{config.multi_starts}.pkl", "wb") as f:
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
