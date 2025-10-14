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
        # heuristic 1, for which it needs to be set to 4 from the Config object (this
        # behavior is implemented in utils/parse_config.py so this variable should
        # not be modified to run the simulations as described in the paper)
        self.multi_starts = 1

        # Horizon and platoon size settings
        self.N = 30  # If commented out, N=15
        # self.num_vehicles = 10  # If commented out, num_vehicles=5

        # Time limits for the primary solvers
        self.extra_opts["gurobi"]["TimeLimit"] = 720
        self.extra_opts["knitro"]["maxtime"] = 720
        self.extra_opts["bonmin"]["time_limit"] = 720
        self.extra_opts["cplex"]["CPXPARAM_TimeLimit"] = 720
        # self.extra_opts["ipopt"]["max_wall_time"] = max_time

        # Backup MINLP MPC parameters
        self.backup_minlp_mip_terminate = 1  # terminate solver after first feasible sol
        self.backup_minlp_maxtime = 600  # maximum time for the backup MPC
