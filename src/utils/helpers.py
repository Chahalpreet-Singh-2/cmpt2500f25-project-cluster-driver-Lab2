from pathlib import Path
import yaml, logging


def resolve_under_root(path: str | Path) -> Path:
    """
    Ensures the given path is resolved relative to the project root.
    Used by lab templates for consistent path handling.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # assume repo root is one level above src/
    root = Path(__file__).resolve().parents[1]
    return (root / p).resolve()
    
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_config(path: Path) -> dict:
    with open(path, "r") as f: return yaml.safe_load(f)

def get_logger(name="ml_project", level=logging.INFO):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)
    return logging.getLogger(name)
