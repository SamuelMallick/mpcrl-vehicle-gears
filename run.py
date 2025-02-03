from agent import Agent, MINLPAgent, HeuristicGearAgent
from dqn_agent import DQNAgent
from env import VehicleTracking
from vehicle import Vehicle
from gymnasium.wrappers import TimeLimit
from utils.wrappers.monitor_episodes import MonitorEpisodes
import numpy as np
from visualisation.plot import plot_evaluation, plot_training
from mpc import (
    HybridTrackingMpc,
    HybridTrackingFuelMpcFixedGear,
    HybridTrackingFuelMpc,
    TrackingMpc,
)

np_random = np.random.default_rng(1)

vehicle = Vehicle()
ep_length = 60
env = MonitorEpisodes(TimeLimit(VehicleTracking(vehicle), max_episode_steps=ep_length))

# mpc = HybridTrackingMpc(5)
# mpc = HybridTrackingFuelMpc(5)
# agent = MINLPAgent(mpc)

mpc = HybridTrackingFuelMpcFixedGear(5)
agent = DQNAgent(mpc, 5, np_random)

# mpc = TrackingMpc(5)
# agent = HeuristicGearAgent(mpc)

# returns, info = agent.evaluate(env, episodes=1)
num_eps = 1000
returns, info = agent.train(
    env,
    episodes=num_eps,
    exp_zero_steps=int(ep_length * num_eps / 2),
    save_freq=50,
    save_path="results",
)
fuel = info["fuel"]
engine_torque = info["T_e"]
engine_speed = info["w_e"]
x_ref = info["x_ref"]
cost = info["cost"]

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)

print(f"cost = {sum(R[0])}")
print(f"fuel = {sum(fuel[0])}")

ep = 0
plot_evaluation(
    x_ref[ep], X[ep], U[ep], R[ep], fuel[ep], engine_torque[ep], engine_speed[ep]
)
plot_training(
    [sum(cost[i]) for i in range(len(cost))],
    [sum(fuel[i]) for i in range(len(fuel))],
    [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
)
