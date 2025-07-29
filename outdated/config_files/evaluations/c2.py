from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "eval_2"
        self.eval_seed = 11
