from config_files.base import ConfigDefault


class EvalBase(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "eval_base"
        self.eval_seed = 0

        # Disable bidirectional RNNs
        # NOTE: this needs to be set to True in case a policy of type c1 or c2 is used
        self.bidirectional = False

        # Number of multi-starts for the optimization
        self.multi_starts = 1

        # Time limits for the solvers
        # self.extra_opts["gurobi"]["TimeLimit"] = 3600
        self.extra_opts["knitro"]["maxtime"] = 3600
        # self.extra_opts["bonmin"]["time_limit"] = max_time
        # self.extra_opts["ipopt"]["max_wall_time"] = max_time
