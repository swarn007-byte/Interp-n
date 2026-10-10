"""Central config loader (YAML + attribute access)."""

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)
