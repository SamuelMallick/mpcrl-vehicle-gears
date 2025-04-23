from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "1_seeds_1"
        self.train_seed = 1
        self.cuda_seed = 1
