"""
Tests for src.utils.helpers module.
"""

from pathlib import Path

from src.utils import helpers


def test_get_project_root_looks_like_repo_root():
    root = helpers.get_project_root()
    # Basic sanity checks: src/ exists under project root
    assert (root / "src").exists()


def test_resolve_under_root_with_relative_path(monkeypatch, tmp_path):
    """
    resolve_under_root('configs/train_config.yaml') should
    prepend the project root returned by get_project_root().
    """
    fake_root = tmp_path / "proj"
    (fake_root / "src").mkdir(parents=True)
    (fake_root / "configs").mkdir()
    cfg = fake_root / "configs" / "train_config.yaml"
    cfg.write_text("train:\n  n_clusters: 3\n")

    # Force helpers.get_project_root() to return our fake root
    monkeypatch.setattr(helpers, "get_project_root", lambda: fake_root)

    resolved = helpers.resolve_under_root("configs/train_config.yaml")
    assert resolved == cfg


def test_resolve_under_root_with_absolute_path(tmp_path):
    """
    If an absolute path is passed, resolve_under_root should return it unchanged
    (or an equivalent Path).
    """
    abs_path = tmp_path / "something" / "file.txt"
    abs_path.parent.mkdir(parents=True)
    abs_path.write_text("hello")

    resolved = helpers.resolve_under_root(abs_path)
    assert resolved == abs_path


def test_load_config_reads_yaml(tmp_path):
    """
    load_config should read a YAML file and return a dict.
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "train:\n"
        "  n_clusters: 7\n"
        "  random_state: 123\n"
    )

    cfg = helpers.load_config(cfg_file)
    assert cfg["train"]["n_clusters"] == 7
    assert cfg["train"]["random_state"] == 123
