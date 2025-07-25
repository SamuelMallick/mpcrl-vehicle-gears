import pickle
import torch
from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "2"

        # initial weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_state_dict = torch.load(
            f"results/1_exp_bug_fix_2/policy_net_step_4325000.pth",
            weights_only=True,
            map_location=device,
        )
        with open(f"results/1_exp_bug_fix_2/data_step_4325000.pkl", "rb") as f:
            data = pickle.load(f)
        self.init_normalization = data["normalization"]

        # penalties
        self.infeas_pen = 0
        self.rl_reward = -1e2

        self.eps_start = 0
        self.use_heuristic = True
