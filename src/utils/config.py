"""Configuration loader utility."""
import json
import yaml
from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    """Returns absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_dataset_profile() -> dict:
    """Read archive and diagnostic settings kept with the local dataset."""
    path = get_project_root() / "data" / "dataset_profile.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def dataset_sample(key: str) -> Path:
    path = load_dataset_profile().get("samples", {}).get(key)
    if not path:
        raise ValueError(f"Set sample '{key}' in data/dataset_profile.json before running this check.")
    return get_project_root() / path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration yaml file.
    
    Args:
        config_path: Path to config.yaml. Defaults to configs/config.yaml in project root.
        
    Returns:
        Dictionary with parsed configuration.
    """
    if config_path is None:
        root = get_project_root()
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = get_project_root() / config_path
            
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

