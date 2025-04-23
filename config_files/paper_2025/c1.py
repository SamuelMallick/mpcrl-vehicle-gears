from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "paper_2025_1"

        self.ep_len = 100
        self.trajectory_type = "type_2"
        self.finite_episodes = True
        self. extra_opts = {
            "gurobi": {},
            "knitro": {},
            "bonmin": {},
            "ipopt": {},
        }
        self.bidirectional = False
