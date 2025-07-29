import pickle

import torch

from config_files.train_c2_seeds.c2_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c2_seed7"
        self.train_seed = 0
        self.cuda_seed = 3653403230

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_state_dict = torch.load(
            "results/c1_baseline/policy_net_step_4325000.pth",
            weights_only=True,
            map_location=device,
        )
        with open("results/c1_baseline/data_step_4325000.pkl", "rb") as f:
            data = pickle.load(f)
        self.init_normalization = data["normalization"]
