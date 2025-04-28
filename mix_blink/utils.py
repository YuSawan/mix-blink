import argparse
import os

import yaml


def load_config_as_namespace(config_file: str | os.PathLike) -> argparse.Namespace:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)
