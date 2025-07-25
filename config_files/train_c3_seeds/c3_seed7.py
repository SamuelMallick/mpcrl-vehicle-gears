from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed7"
        self.train_seed = 7
        self.cuda_seed = 7
        self.bidirectional = False
