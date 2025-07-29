from config_files.train_c2_seeds.c2_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c2_seed9"
        self.train_seed = 19
        self.cuda_seed = 19
