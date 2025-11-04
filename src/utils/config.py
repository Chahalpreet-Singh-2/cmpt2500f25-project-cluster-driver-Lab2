# src/utils/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

# ---------- Paths ----------

def get_project_root() -> Path:
    """
    Returns the project root (folder that contains src/, configs/, data/, etc.).
    We locate this file (src/utils/config.py) and go up two levels.
    """
    return Path(__file__).resolve().parents[2]

# ---------- YAML I/O ----------

def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env_vars(data)

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML config relative to project root if a relative path is provided.
    Expands ${ENV_VAR} in strings.
    """
    root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return _read_yaml(p)

def save_config(config: Mapping[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=True)

# ---------- Overrides & merging ----------

def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge 'override' into 'base'.
    Lists and scalars are replaced; dicts are merged.
    """
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base

def with_overrides(config_path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """
    Load a YAML config and apply CLI overrides:
        cfg = with_overrides("configs/train_config.yaml", {"train": {"n_clusters": 12}})
    """
    cfg = load_config(config_path)
    if overrides:
        deep_update(cfg, overrides)
    return cfg

# ---------- Small helpers ----------

def resolve_under_root(relative_or_abs_path: str | Path) -> Path:
    """
    Convert a relative path (e.g., 'data/processed/file.parquet') into an absolute
    path under the project root. If already absolute, return as-is.
    """
    p = Path(relative_or_abs_path)
    return p if p.is_absolute() else (get_project_root() / p).resolve()

def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand ${VAR} in strings using environment variables.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    return obj
