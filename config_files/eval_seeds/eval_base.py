from config_files.base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "eval_base"
        self.eval_seed = 1312

        # Toggle use of bidirectional RNNs
        # NOTE: needs to be set True if policy of type c1 or c2 is used
        self.bidirectional = False

        # Number of multi-starts for the optimization
        # NOTE: currently multi-starting is hardcoded in the agent files except for
        # heuristic 1, for which it needs to be set to 4 from the Config object
        self.multi_starts = 1

        # Time limits for the primary solvers
        self.extra_opts["gurobi"]["TimeLimit"] = 3600
        self.extra_opts["knitro"]["maxtime"] = 3600
        # self.extra_opts["bonmin"]["time_limit"] = max_time
        # self.extra_opts["ipopt"]["max_wall_time"] = max_time

        # Backup MINLP MPC parameters
        self.backup_minlp_mip_terminate = 1  # terminate solver after first feasible sol
        self.backup_minlp_maxtime = 60  # maximum time for the backup MPC
