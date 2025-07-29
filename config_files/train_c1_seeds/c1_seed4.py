from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c1_seed4"
        self.train_seed = 4
        self.cuda_seed = 4
