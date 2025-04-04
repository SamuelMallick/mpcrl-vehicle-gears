class ConfigDefault:
    id = "base"

    # -----------general parameters----------------
    N = 15
    ep_len = 100
    num_eps = 50000
    trajectory_type = "type_3"
    windy = False
    save_every_episode = False
    infinite_episodes = False
    max_steps = ep_len * num_eps
    multi_starts = 1

    # -----------network parameters----------------
    # initial weights
    init_state_dict = {}
    init_normalization = ()

    # hyperparameters
    gamma = 0.9
    learning_rate = 0.001
    tau = 0.001

    # archticeture
    clip = False
    n_states = 8
    n_hidden = 256
    n_actions = 3
    n_layers = 4
    bidirectional = True
    normalize = True

    # exploration
    eps_start = 0.99
    esp_zero_steps = int(ep_len * num_eps / 2)

    # penalties
    clip_pen = 0
    infeas_pen = 1e4
    rl_reward = -1e2

    # memory
    memory_size = 100000
    batch_size = 128

    max_grad = 100
