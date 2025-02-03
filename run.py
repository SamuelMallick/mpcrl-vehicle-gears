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
    HybridTrackingMpcFixedGear,
    HybridTrackingFuelMpc,
    TrackingMpc,
)

np_random = np.random.default_rng(1)

vehicle = Vehicle()
env = MonitorEpisodes(TimeLimit(VehicleTracking(vehicle), max_episode_steps=60))

# mpc = HybridTrackingMpc(5)
# mpc = HybridTrackingFuelMpc(5)
# agent = MINLPAgent(mpc)

mpc = HybridTrackingMpcFixedGear(5)
agent = DQNAgent(mpc, 5, np_random)

# mpc = TrackingMpc(5)
# agent = HeuristicGearAgent(mpc)

# returns, info = agent.evaluate(env, episodes=1)
returns, info = agent.train(env, episodes=10)
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


# plot_evaluation(x_ref[0], X[0], U[0], R[0], fuel[0], engine_torque[0], engine_speed[0])
plot_training(
    [sum(cost[i]) for i in range(len(cost))],
    [sum(fuel[i]) for i in range(len(fuel))],
    [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
)
