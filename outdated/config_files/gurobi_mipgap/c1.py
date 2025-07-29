from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "gurobi_mipgap_c1"
        self.extra_opts["gurobi"]["MIPGap"] = 1e-3  # default val is 1e-4
