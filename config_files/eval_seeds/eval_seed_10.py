from config_files.eval_seeds.eval_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "eval_seed_10"
        self.eval_seed = 10

        # NOTE: seed 10 was used during the training of some of the policies. It is used
        # here only to replicate some old results that were obtained with this seed, as
        # a sanity check. Results obtained from this seed should not be used for
        # policy evaluation purposes.
