import pickle

import torch

from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c2_base"
        self.train_seed = 0
        self.cuda_seed = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_state_dict = torch.load(
            "results/c1/c1_seed1/policy_net_step_5000000.pth",
            weights_only=True,
            map_location=device,
        )
        with open("results/c1/c1_seed1/data_step_5000000.pkl", "rb") as f:
            data = pickle.load(f)
        self.init_normalization = data["normalization"]

        # penalties
        self.infeas_pen = 0
        self.rl_reward = -1e2

        self.eps_start = 0
        self.use_heuristic = True
