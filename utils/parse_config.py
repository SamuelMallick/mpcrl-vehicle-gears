from config_files.eval_seeds.eval_base import EvalBase


def parse_config_eval(cmd_args) -> EvalBase:
    """
    Parse command line arguments and generates a config object.

    Args:
        cmd_args (list): List of command line arguments.

    Returns:
        Config: Config object with parameters set based on command line arguments.
    """

    default_config_file = EvalBase()
