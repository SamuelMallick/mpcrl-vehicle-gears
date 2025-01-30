from agent import Agent, MINLPAgent, HeuristicGearAgent
from env import VehicleTracking
from vehicle import Vehicle
from gymnasium.wrappers import TimeLimit
from utils.wrappers.monitor_episodes import MonitorEpisodes
import matplotlib.pyplot as plt
import numpy as np
from visualisation.plot import plot
from mpc import HybridTrackingMpc, HybridTrackingMpcFixedGear, HybridTrackingFuelMpc


vehicle = Vehicle()
env = MonitorEpisodes(TimeLimit(VehicleTracking(vehicle), max_episode_steps=50))

# mpc = HybridTrackingMpc(5)
mpc = HybridTrackingFuelMpc(5)
agent = MINLPAgent(mpc)

# mpc = HybridTrackingMpcFixedGear(5)
# agent = HeuristicGearAgent(mpc)

returns, info = agent.evaluate(env, episodes=1)
fuel = info["fuel"]
engine_torque = info["T_e"]
engine_speed = info["w_e"]
x_ref = info["x_ref"]

X = list(env.observations)
U = list(env.actions)
R = list(env.rewards)

# plot first episode
plot(x_ref[0], X[0], U[0], R[0], fuel[0], engine_torque[0], engine_speed[0])
