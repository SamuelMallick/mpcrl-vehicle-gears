from config_files.c2_seeds.c2_base import ConfigC2Base


class Config(ConfigC2Base):

    def __init__(self):
        super().__init__()
        self.id = "c2_seed1"
        self.train_seed = 11
        self.cuda_seed = 11
