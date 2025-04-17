from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "multi_starts_1"
        self.multi_starts = 1
