import os
import sys
from mpc import HybridTrackingMpc
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import os, sys

sys.path.append(os.getcwd())
from config_files.base import ConfigDefault


class Config(ConfigDefault):
    id = "6"  # base but with no expert mpc

    # -----------general parameters----------------
    N = 5
    ep_len = 100
    num_eps = 50000
    trajectory_type = "type_3"

    # -----------network parameters----------------
    # hyperparameters
    gamma = 0.9
    learning_rate = 0.001
    tau = 0.001

    # archticeture
    n_states = 7
    n_hidden = 64
    n_actions = 3
    n_layers = 2
    bidirectional = True
    normalize = False

    # exploration
    eps_start = 0.99
    esp_zero_steps = int(ep_len * num_eps / 2)

    # penalties
    clip_pen = 0
    infeas_pen = 1e4

    # memory
    memory_size = 100000
    batch_size = 128

    expert_mpc = None
