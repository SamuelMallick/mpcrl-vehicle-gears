from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "3_seeds_5"
        self.train_seed = 5
        self.cuda_seed = 5
        self.bidirectional = False
