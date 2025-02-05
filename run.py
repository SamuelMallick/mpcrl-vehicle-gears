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

np_random = np.random.default_rng(1)

vehicle = Vehicle()
ep_length = 60
env = MonitorEpisodes(TimeLimit(VehicleTracking(vehicle), max_episode_steps=ep_length))

# mpc = SolverTimeRecorder(
#     HybridTrackingMpc(
#         5, optimize_fuel=True, convexify_fuel=False, convexify_dynamics=False
#     )
# )
# agent = MINLPAgent(mpc)
# mpc = SolverTimeRecorder(TrackingMpc(5))
# agent = HeuristicGearAgent(mpc)

# returns, info = agent.evaluate(env, episodes=1)

mpc = SolverTimeRecorder(HybridTrackingFuelMpcFixedGear(5, convexify_fuel=False))
agent = DQNAgent(mpc, 5, np_random)
state_dict = torch.load(
    "results/many_traj_N_5/policy_net_ep_49000.pth",
    weights_only=True,
    map_location="cpu",
)

returns, info = agent.evaluate(env, episodes=1, policy_net_state_dict=state_dict)
# num_eps = 1
# returns, info = agent.train(
#     env,
#     episodes=num_eps,
#     exp_zero_steps=int(ep_length * num_eps / 2),
#     save_freq=50,
#     save_path="results",
#     seed=0,
# )
fuel = info["fuel"]
engine_torque = info["T_e"]
engine_speed = info["w_e"]
x_ref = info["x_ref"]
# cost = info["cost"]

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)

print(f"cost = {sum(R[0])}")
print(f"fuel = {sum(fuel[0])}")
print(f"total mpc solve times = {sum(mpc.solver_time)}")

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
