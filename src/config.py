
from pathlib import Path
from typing import Any, Dict, Union

import yaml


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "config.yaml"
)


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise RuntimeError("Configuration file is empty.")

    return cfg


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def get(cfg: Dict[str, Any], *keys, default=None):
    """
    Safely access nested config values.

    Example
    -------
    lr = get(cfg, "training", "learning_rate")

    instead of

    lr = cfg["training"]["learning_rate"]
    """

    value = cfg

    for key in keys:
        if not isinstance(value, dict):
            return default

        value = value.get(key)

        if value is None:
            return default

    return value


# ---------------------------------------------------------------------
# Pretty Printing
# ---------------------------------------------------------------------

def print_config(cfg: Dict[str, Any]) -> None:
    """
    Nicely print configuration.
    """

    print("=" * 70)
    print("Loaded Configuration")
    print("=" * 70)

    print(
        yaml.dump(
            cfg,
            default_flow_style=False,
            sort_keys=False,
        )
    )

    print("=" * 70)


# ---------------------------------------------------------------------
# Load global config
# ---------------------------------------------------------------------

cfg = load_config()


# ---------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print_config(cfg)