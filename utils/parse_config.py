import os
import sys
import argparse
import importlib.util
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from config_files.base import Config


def parse_config(cmd_args: list[str]) -> "Config":
    """
    Parse command line arguments and generate a config object.

    Args:
        cmd_args (list): List of command line arguments.

    Returns:
        Config object with parameters set based on command line arguments.
    """

    parser = argparse.ArgumentParser(description="Parse evaluation configuration.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="eval",
        help="Mode of operation: 'train' or 'eval'.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file to use. The config file must be in config_files, and "
        "its path under config_files/ must be provided "
        "(e.g., eval_seeds/eval_base.py).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="train_c4/c4_seed4",
        help=[
            "Name of the learning-based policy to use when running l_mpc.py. The path "
            "must include any subfolder of results/."
        ],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use (for reproducibility).",
    )

    parser.add_argument(
        "--max-time",
        type=int,
        default=None,
        help="Maximum time limit for the solver in seconds.",
    )

    args = parser.parse_args()
    script_name = cmd_args[0].split("/")[-1]

    # Load the configuration based on the provided config file
    config = load_config_from_file(args.config, args.mode)

    # Update config properties based on command line arguments
    if script_name == "l_mpc.py":
        policy_folder = args.policy.split("/")[0]
        policy_name = args.policy.split("/")[-1]
        match policy_name[:2]:
            case "c1" | "c2":
                config.bidirectional = True
            case "c3" | "c4":
                config.bidirectional = False

        for filename in os.listdir(f"results/{policy_folder}/{policy_name}"):
            if filename.endswith(".pth"):
                config.policy_filename = f"{policy_folder}/{policy_name}/{filename}"
            elif filename.endswith(".pkl"):
                config.normalization_data_filename = (
                    f"{policy_folder}/{policy_name}/{filename}"
                )
        config.results_folder_name = f"eval_l_mpc/{policy_name}"

    elif script_name == "heuristic_mpc_1.py":
        config.multi_starts = 4  # heuristic 1 uses 4 multi-starts

    if args.seed is not None:
        if args.mode == "eval":
            config.eval_seed = args.seed
            config.id = f"eval_seed_{args.seed}"
        elif args.mode == "train":
            config.train_seed = args.seed
            config.cuda_seed = args.seed
            config_base_name = config.id[0:2]
            config.id = f"{config_base_name}_seed_{args.seed}"

    if args.max_time is not None:
        config.extra_opts["gurobi"]["TimeLimit"] = args.max_time
        config.extra_opts["knitro"]["maxtime"] = args.max_time
        # config.extra_opts["ipopt"]["max_wall_time"] = args.max_time

    return config


def load_config_from_file(filename: str, mode: str) -> "Config":
    """
    Create an instance of the Config class from a given file.

    Args:
        filename (str): The path to the configuration file starting from config_files/
            (e.g., "config_files/eval_seeds/eval_base.py").
        mode (str): The script running mode {"train", "eval"}.

    Returns:
        Config: An instance of the Config class.
    """

    # Check if config file is provided, if not, use default based on mode
    match mode:
        case "train":
            if filename is None:
                filename = "config_files/train_c3_seeds/c3_seed1.py"
            else:
                filename = os.path.join("config_files/", filename)
        case "eval":
            if filename is None:
                filename = "config_files/eval_seeds/eval_base.py"
            else:
                filename = os.path.join("config_files/", filename)

    # Get the absolute path of the config file and the module name
    module_path = os.path.abspath(filename)
    module_name = os.path.splitext(os.path.basename(filename))[0]

    # Create a module object and load it
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Retrieve the class
    cls = getattr(module, "Config", None)
    if cls is None:
        raise AttributeError(f"Class 'Config' not found in {module_path}")

    return cls()
