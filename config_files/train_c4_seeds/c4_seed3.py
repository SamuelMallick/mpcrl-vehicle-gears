from config_files.c4_seeds.c4_base import ConfigC4Base


class Config(ConfigC4Base):

    def __init__(self):
        super().__init__()
        self.id = "c4_seed3"
        self.train_seed = 13
        self.cuda_seed = 13
