from config_files.train_c2_seeds.c2_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c2_seed3"
        self.train_seed = 13
        self.cuda_seed = 13
