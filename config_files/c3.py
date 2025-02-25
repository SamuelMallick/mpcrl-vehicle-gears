import os
import sys

import torch
from mpc import HybridTrackingMpc
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import os, sys

sys.path.append(os.getcwd())
from config_files.base import ConfigDefault


class Config(ConfigDefault):
    id = "3"  # 2 but with init state dict from sl

    # -----------general parameters----------------
    N = 15
    ep_len = 100
    num_eps = 50000
    trajectory_type = "type_2"

    # -----------network parameters----------------
    # initial weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_state_dict = torch.load(
        f"results/sl_data/shift/policy_net_ep_300_epoch_2400.pth",
        weights_only=True,
        map_location=device,
    )

    # hyperparameters
    gamma = 0.9
    learning_rate = 0.0001
    tau = 0.001

    # archticeture
    n_hidden = 256
    n_actions = 3
    n_layers = 4
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
