from config_files.base import ConfigDefault


class EvalBase(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "eval_base"
        self.eval_seed = 0

        self.multi_starts = 1  # Should this be changed?

        max_time = None  # None or a positive float value
        if max_time is not None:
            self.extra_opts["gurobi"]["TimeLimit"] = max_time
            self.extra_opts["knitro"]["maxtime"] = max_time
            self.extra_opts["bonmin"]["time_limit"] = max_time
            self.extra_opts["ipopt"]["max_wall_time"] = max_time
