from config_files.base import ConfigDefault


class Config(ConfigDefault):
    id = "2"

    # -----------general parameters----------------
    ep_len = 1000
    max_time = None
    inter_vehicle_distance = 100
