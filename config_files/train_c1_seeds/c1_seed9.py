from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c1_seed9"
        self.train_seed = 9
        self.cuda_seed = 9
