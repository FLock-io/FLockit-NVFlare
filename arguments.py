import argparse
from omegaconf import OmegaConf

def add_args():
    parser = argparse.ArgumentParser(description="FLockit")
    parser.add_argument(
        "--yaml_config_file",
        "--conf",
        help="Templates configuration file (.yaml)",
        type=str,
        default="",
    )

    args, unknown = parser.parse_known_args()
    return args

def load_arguments():
    cmd_args = add_args()
    args = OmegaConf.load(cmd_args.yaml_config_file)

    return args