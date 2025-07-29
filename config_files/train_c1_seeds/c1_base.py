from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c1_base"
        self.train_seed = 0
        self.cuda_seed = 0
