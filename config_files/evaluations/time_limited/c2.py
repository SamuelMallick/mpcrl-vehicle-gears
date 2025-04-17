from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "eval_time_limited_2"
        self.eval_seed = 11
        max_time = 1
        self.extra_opts["gurobi"]["TimeLimit"] = max_time
        self.extra_opts["knitro"]["maxtime"] = max_time
        self.extra_opts["bonmin"]["time_limit"] = max_time
        self.extra_opts["ipopt"]["max_wall_time"] = max_time
