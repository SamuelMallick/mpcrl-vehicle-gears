from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c1_seed3"
        self.train_seed = 3
        self.cuda_seed = 3
