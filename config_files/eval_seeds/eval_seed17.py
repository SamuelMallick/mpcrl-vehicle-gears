from config_files.eval_seeds.eval_base import EvalBase


class Config(EvalBase):

    def __init__(self):
        super().__init__()
        self.id = "eval_seed17"
        self.eval_seed = 117
