from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed8"
        self.train_seed = 8
        self.cuda_seed = 8
        self.bidirectional = False
