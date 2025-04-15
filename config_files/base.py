class ConfigDefault:
    id = "base"

    SAVE = True
    PLOT = False

    # -----------general parameters----------------
    N = 15
    ep_len = 1000
    max_train_steps = 100 * 50000
    max_episodes = 50000
    trajectory_type = "type_3"
    windy = False
    save_every_episode = False
    finite_episodes = False
    terminate_on_distance = True
    inter_vehicle_distance = 25
    eval_seed = 10
    train_seed = 0
    num_vehicles = 5

    # -----------solver parameters----------------
    multi_starts = 1
    extra_opts = {
        "gurobi": {"MIPGap": 1e-9},
        "knitro": {},
        "bonmin": {},
        "ipopt": {},
    }
    max_time = None
    if max_time is not None:
        extra_opts["gurobi"]["TimeLimit"] = max_time
        extra_opts["knitro"]["maxtime"] = max_time
        extra_opts["bonmin"]["time_limit"] = max_time
        extra_opts["ipopt"]["max_wall_time"] = max_time

    # -----------network parameters----------------
    # initial weights
    init_state_dict = {}
    init_normalization = ()

    # hyperparameters
    gamma = 0.9
    learning_rate = 0.001
    tau = 0.001

    # archticeture
    n_states = 8
    n_hidden = 256
    n_actions = 3
    n_layers = 4
    bidirectional = True
    normalize = True

    # exploration
    eps_start = 0.99
    exp_zero_steps = int(max_train_steps / 2)

    # penalties
    infeas_pen = 1e4
    rl_reward = 0  # -1e2

    # memory
    memory_size = 100000
    batch_size = 128

    max_grad = 100
