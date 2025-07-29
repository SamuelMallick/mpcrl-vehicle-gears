from config_files.eval_seeds.eval_base import Config


class Config(Config):

    def __init__(self):
        super().__init__()
        self.id = "eval_time_1s"
        self.eval_seed = 10

        # Set time limit for the solvers to 1 second
        self.extra_opts["gurobi"]["TimeLimit"] = 1
        self.extra_opts["knitro"]["maxtime"] = 1
        # self.extra_opts["bonmin"]["time_limit"] = 1
        # self.extra_opts["ipopt"]["max_wall_time"] = 1

        # Backup MINLP MPC parameters
        self.backup_minlp_mip_terminate = 1  # stop at 1st feas sol
        self.backup_minlp_maxtime = 1  # 1s
