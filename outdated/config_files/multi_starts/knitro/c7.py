from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "multi_starts_knitro_7"
        self.multi_starts = 1
        max_time = 50
        self.extra_opts["gurobi"]["TimeLimit"] = max_time
        self.extra_opts["knitro"]["maxtime"] = max_time
        self.extra_opts["bonmin"]["time_limit"] = max_time
        self.extra_opts["ipopt"]["max_wall_time"] = max_time

        self.extra_opts["knitro"]["ms_maxsolves"] = 50
