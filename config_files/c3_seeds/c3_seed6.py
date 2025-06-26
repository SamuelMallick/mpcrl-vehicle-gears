from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed6"
        self.train_seed = 6
        self.cuda_seed = 6
        self.bidirectional = False
