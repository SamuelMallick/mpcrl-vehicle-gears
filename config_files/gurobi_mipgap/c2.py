from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "gurobi_mipgap_c2"
        self.extra_opts["gurobi"]["MIPGap"] = 1e-5
