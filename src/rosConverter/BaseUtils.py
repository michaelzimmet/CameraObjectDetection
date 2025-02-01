from pathlib import Path

import yaml


def read_config_file(file: Path):
    with file.open('r') as f:
        config = yaml.safe_load(f)
    return config