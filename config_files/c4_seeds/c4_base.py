import pickle

import torch

from config_files.base import ConfigDefault


class ConfigC4Base(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c4_base"
        self.train_seed = 1
        self.cuda_seed = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_state_dict = torch.load(
            "results/c3/c3_seed1/policy_net_step_5000000.pth",
            weights_only=True,
            map_location=device,
        )
        with open("results/c3/c3_seed1/data_step_5000000.pkl", "rb") as f:
            data = pickle.load(f)
        self.init_normalization = data["normalization"]

        # penalties
        self.infeas_pen = 0
        self.rl_reward = -1e2

        self.eps_start = 0
        self.use_heuristic = True

        # Disable bidirectional RNN
        self.bidirectional = False
