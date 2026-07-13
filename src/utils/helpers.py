import logging
import os
import json
import yaml
from typing import Any, Dict

def setup_logging(level: str = "INFO") -> None:
    """Configures the root logger with formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML config file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_file(file_path: str) -> str:
    """Loads the raw content of a text file (e.g. system prompt or template)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_json(file_path: str, data: Any) -> None:
    """Saves data into a JSON file, ensuring the parent directory exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
