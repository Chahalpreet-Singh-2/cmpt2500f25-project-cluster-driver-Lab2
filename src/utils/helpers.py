from pathlib import Path
import yaml, logging

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config(path: Path) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def get_logger(name="ml_project", level=logging.INFO):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)
    return logging.getLogger(name)
