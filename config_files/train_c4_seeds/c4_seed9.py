from config_files.train_c4_seeds.c4_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c4_seed9"
        self.train_seed = 19
        self.cuda_seed = 19
