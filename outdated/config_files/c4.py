from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "4"
        self.infeas_pen = 1e3
