from config_files.eval_seeds.eval_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "eval_time_1s"
        self.eval_seed = 10

        # Set time limit for the solvers to 1 second

        # For single vehicle evaluation (M=1)
        # self.extra_opts["gurobi"]["TimeLimit"] = 1
        # self.extra_opts["knitro"]["maxtime"] = 1
        # self.extra_opts["ipopt"]["max_wall_time"] = 1

        # For platooning evaluation (M=5)
        self.extra_opts["gurobi"]["TimeLimit"] = 0.2
        self.extra_opts["knitro"]["maxtime"] = 0.2
        self.extra_opts["ipopt"]["max_wall_time"] = 0.2
        self.extra_opts["cplex"]["CPXPARAM_TimeLimit"] = 0.2

        # Backup MINLP MPC parameters
        self.backup_minlp_mip_terminate = 1  # stop at 1st feas sol
        self.backup_minlp_maxtime = 600  # [s]
