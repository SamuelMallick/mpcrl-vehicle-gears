from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "multi_starts_4"
        self.multi_starts = 100
