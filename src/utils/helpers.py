from pathlib import Path
import yaml, logging


def resolve_under_root(relative_path: str):
    """Return absolute path under the project root."""
    # If you already have get_project_root() here, reuse it.
    try:
        root = get_project_root()
    except NameError:
        # Fallback: helpers.py is in src/utils/, so project root is two levels up
        root = Path(__file__).resolve().parents[2]
    return Path(root) / relative_path
    
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config(path: Path) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def get_logger(name="ml_project", level=logging.INFO):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)
    return logging.getLogger(name)
