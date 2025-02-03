import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys, os

sys.path.append(os.getcwd())
from visualisation.plot import plot_training

file_name = "results/data_ep_50.pkl"
with open(file_name, "rb") as f:
    data = pickle.load(f)

cost = data["cost"]
fuel = data["fuel"]
R = data["R"]

plot_training(
    [sum(cost[i]) for i in range(len(cost))],
    [sum(fuel[i]) for i in range(len(fuel))],
    [sum(R[i]) - sum(fuel[i]) for i in range(len(R))],
    [sum(cost[i]) - sum(R[i]) for i in range(len(R))],
)
