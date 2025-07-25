from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed1"
        self.train_seed = 1
        self.cuda_seed = 1
        self.bidirectional = False
