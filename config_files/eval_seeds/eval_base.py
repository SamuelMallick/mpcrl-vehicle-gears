from config_files.base import ConfigDefault


class EvalBase(ConfigDefault):

    def __init__(self):
        super().__init__()
        self.id = "eval_base"
        self.eval_seed = 0

        # Toggle for use of bidirectional RNN
        # NOTE: This value must be kept constant during a set of experiments, i.e., a
        # batch of experiments that runs at the same time must be either formed only by
        # experiments with this value set to True or only by experiments with this value
        # set to False.
        self.bidirectional = False

        # Number of multi-starts for the optimization
        # self.multi_starts = 1

        # Time limits for the solvers
        # max_time = None
        # self.extra_opts["gurobi"]["TimeLimit"] = max_time
        self.extra_opts["knitro"]["maxtime"] = 180
        # self.extra_opts["bonmin"]["time_limit"] = max_time
        # self.extra_opts["ipopt"]["max_wall_time"] = max_time
