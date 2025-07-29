from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed3"
        self.train_seed = 3
        self.cuda_seed = 3
        self.bidirectional = False
