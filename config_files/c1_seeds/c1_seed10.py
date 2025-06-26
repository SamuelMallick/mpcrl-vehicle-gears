from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c1_seed10"
        self.train_seed = 10
        self.cuda_seed = 10
