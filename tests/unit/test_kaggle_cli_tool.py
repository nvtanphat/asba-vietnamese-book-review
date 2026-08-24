from __future__ import annotations

import json
from pathlib import Path

from tools.kaggle_cli import cli


def test_prepare_data_stages_flat_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "data/raw").mkdir(parents=True)
    (root / "data/splits").mkdir(parents=True)
    (root / "data/maps").mkdir(parents=True)
    (root / "data/raw/tiki-book-review_merged_fixed_v3.json").write_text("[]", encoding="utf-8")
    (root / "data/splits/train.json").write_text("[]", encoding="utf-8")
    (root / "data/maps/emoji_map.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")

    stage = cli.stage_data_dataset("alice/sentenai-data")
    md = json.loads((stage / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert md["id"] == "alice/sentenai-data"
    assert md["licenses"][0]["name"] == "unknown"
    assert (stage / "tiki-book-review_merged_fixed_v3.json").exists()
    assert (stage / "train.json").exists()
    assert (stage / "emoji_map.json").exists()


def test_generated_kaggle_runner_is_valid_python():
    spec = cli.KernelSpec(
        model="phobert",
        kernel_handle="alice/sentenai-phobert",
        data_handle="alice/sentenai-data",
        resume_handle="alice/sentenai-phobert-resume",
    )
    source = cli.kaggle_runner_source(spec)
    compile(source, "run.py", "exec")
    assert "--resume" in source
    assert "--no-test" in source
    assert "NvidiaTeslaT4" not in source  # accelerator belongs to metadata/push, not training code


def test_prepare_kernel_has_t4_and_sources(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "WORK_ROOT", tmp_path / "work")
    spec = cli.KernelSpec(
        model="xlmr",
        kernel_handle="alice/sentenai-xlmr",
        data_handle="alice/sentenai-data",
        resume_handle="alice/sentenai-xlmr-resume",
    )
    stage = cli.prepare_kernel(spec)
    md = json.loads((stage / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert md["machine_shape"] == "NvidiaTeslaT4"
    assert md["dataset_sources"] == ["alice/sentenai-data", "alice/sentenai-xlmr-resume"]
    assert md["kernel_type"] == "script"
    assert (stage / "sentenai_src.zip").exists()
