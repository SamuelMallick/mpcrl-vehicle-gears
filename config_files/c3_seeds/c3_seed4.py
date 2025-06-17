from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "c3_seed4"
        self.train_seed = 4
        self.cuda_seed = 4
        self.bidirectional = False
