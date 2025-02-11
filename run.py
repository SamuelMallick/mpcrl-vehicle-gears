from agent import Agent, MINLPAgent, HeuristicGearAgent
from dqn_agent import DQNAgent
from env import VehicleTracking
from vehicle import Vehicle
from gymnasium.wrappers import TimeLimit
from utils.wrappers.monitor_episodes import MonitorEpisodes
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import numpy as np
from visualisation.plot import plot_evaluation, plot_training
from mpc import (
    HybridTrackingMpc,
    HybridTrackingFuelMpcFixedGear,
    TrackingMpc,
)
import torch
import pickle
from typing import Literal

SAVE = True
PLOT = True

sim_type: Literal[
    "rl_mpc_train", "rl_mpc_eval", "miqp_mpc", "minlp_mpc", "heuristic_mpc"
] = "rl_mpc_train"


vehicle = Vehicle()
ep_length = 100
num_eval_eps = 100
N = 5
seed = 0
env = MonitorEpisodes(TimeLimit(VehicleTracking(vehicle), max_episode_steps=ep_length))
np_random = np.random.default_rng(seed)

if sim_type == "rl_mpc_train" or sim_type == "rl_mpc_eval":
    expert_mpc = SolverTimeRecorder(HybridTrackingMpc(N, optimize_fuel=True))
    mpc = SolverTimeRecorder(HybridTrackingFuelMpcFixedGear(N, convexify_fuel=False))
    agent = DQNAgent(mpc, N, np_random, expert_mpc=expert_mpc)
    if sim_type == "rl_mpc_train":
        policy_state_dict = torch.load(
            "results/N_5/policy_net_ep_49999.pth",
            weights_only=True,
        )
        target_state_dict = torch.load(
            "results/N_5/target_net_ep_49999.pth",
            weights_only=True,
        )
        with open("results/N_5/data_ep_49999.pkl", "rb") as f:
            info = pickle.load(f)
        num_eps = 100000
        returns, info = agent.train(
            env,
            episodes=num_eps,
            exp_zero_steps=int(ep_length * 50000 / 2),
            save_freq=1000,
            save_path="results/N_5",
            seed=seed,
            policy_net_state_dict=policy_state_dict,
            target_net_state_dict=target_state_dict,
            info_dict=info,
            start_episode=50000,
            start_exp_step=int(ep_length * 50000),
        )
    else:
        state_dict = torch.load(
            "results/many_traj_N_5/policy_net_ep_49000.pth",
            weights_only=True,
            map_location="cpu",
        )
        returns, info = agent.evaluate(
            env, episodes=num_eval_eps, policy_net_state_dict=state_dict, seed=seed
        )

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
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=seed)
elif sim_type == "heuristic_mpc":
    mpc = SolverTimeRecorder(TrackingMpc(N))
    agent = HeuristicGearAgent(mpc)
    returns, info = agent.evaluate(env, episodes=num_eval_eps, seed=seed)


fuel = info["fuel"]
engine_torque = info["T_e"]
engine_speed = info["w_e"]
x_ref = info["x_ref"]
if sim_type == "rl_mpc_train":
    cost = info["cost"]

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)

print(f"average cost = {sum([sum(R[i]) for i in range(len(R))]) / len(R)}")
print(f"average fuel = {sum([sum(fuel[i]) for i in range(len(fuel))]) / len(fuel)}")
print(f"total mpc solve times = {sum(mpc.solver_time)}")

if SAVE:
    with open(f"results/evaluations/{sim_type}_N_{N}.pkl", "wb") as f:
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
            },
            f,
        )

if PLOT:
    ep = 0
    plot_evaluation(
        x_ref[ep], X[ep], U[ep], R[ep], fuel[ep], engine_torque[ep], engine_speed[ep]
    )
    # plot_training(
    #     [sum(cost[i]) for i in range(len(cost))],
    #     [sum(fuel[i]) for i in range(len(fuel))],
    #     [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    #     [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
    # )
