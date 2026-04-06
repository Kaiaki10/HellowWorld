import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent / ".env")

# Load settings.yaml
_settings_path = Path(__file__).parent / "settings.yaml"
with open(_settings_path, "r") as f:
    settings = yaml.safe_load(f)


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_setting(*keys, default=None):
    """Retrieve a nested setting by key path. e.g. get_setting('risk', 'kelly_fraction')"""
    value = settings
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
        else:
            return default
    return value
