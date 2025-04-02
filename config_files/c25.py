import os
import sys
from mpc import HybridTrackingMpc
from utils.wrappers.solver_time_recorder import SolverTimeRecorder
import os, sys

sys.path.append(os.getcwd())
from config_files.base import ConfigDefault


class Config(ConfigDefault):
    id = "25"  # 11 but with inf episodes and aggressive traj

    # -----------general parameters----------------
    N = 15
    ep_len = 1000
    num_eps = 50000
    trajectory_type = "type_3"
    infinite_episodes = True
    max_steps = 100 * 50000
    multi_starts = 1

    # -----------network parameters----------------
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
