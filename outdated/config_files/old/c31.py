import os
import pickle
import sys
from mpcs.mpc import HybridTrackingMpc
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import os, sys
import torch

sys.path.append(os.getcwd())
from config_files.base import Config


class Config(Config):
    id = "31"  # 30 but no exp

    # -----------general parameters----------------
    N = 15
    ep_len = 1000
    num_eps = 50000
    trajectory_type = "type_3"
    infinite_episodes = True
    max_steps = 100 * 50000
    multi_starts = 1

    # -----------network parameters----------------
    # initial weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_state_dict = torch.load(
        f"dev/results/25/policy_net_step_4050000.pth",
        weights_only=True,
        map_location=device,
    )
    with open(f"dev/results/25/data_step_4050000.pkl", "rb") as f:
        data = pickle.load(f)
    init_normalization = data["normalization"]

    # hyperparameters
    gamma = 0.9
    learning_rate = 0.0001
    tau = 0.001

    # archticeture
    n_hidden = 256
    n_actions = 3
    n_layers = 4
    bidirectional = True
    normalize = True

    # exploration
    eps_start = 0
    esp_zero_steps = int(ep_len * num_eps / 2)

    # penalties
    clip_pen = 0
    infeas_pen = 1e4

    # memory
    memory_size = 100000
    batch_size = 128

    def __init__(self, sim_type: str):
        # used for generating gears at first time step of episodes
        if sim_type == "minlp_mpc":
            self.expert_mpc = None
        else:
            self.expert_mpc = SolverTimeRecorder(
                HybridTrackingMpc(
                    self.N,
                    optimize_fuel=True,
                    convexify_fuel=True,
                    convexify_dynamics=True,
                )
            )
