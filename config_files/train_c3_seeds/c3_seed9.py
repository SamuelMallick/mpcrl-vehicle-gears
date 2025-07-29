from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed9"
        self.train_seed = 9
        self.cuda_seed = 9
        self.bidirectional = False
