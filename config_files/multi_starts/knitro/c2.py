from config_files.base import ConfigDefault


class Config(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "multi_starts_knitro_2"
        self.multi_starts = 5
        max_time = 5
        self.extra_opts["gurobi"]["TimeLimit"] = max_time
        self.extra_opts["knitro"]["maxtime"] = max_time
        self.extra_opts["bonmin"]["time_limit"] = max_time
        self.extra_opts["ipopt"]["max_wall_time"] = max_time
