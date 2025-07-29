from config_files.train_c4_seeds.c4_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "c4_seed1"
        self.train_seed = 11
        self.cuda_seed = 11

        # Sanity check for bidirectional setting
        if self.bidirectional:
            raise ValueError(
                "Bidirectional setting should be set to False in C4 configuration."
            )
