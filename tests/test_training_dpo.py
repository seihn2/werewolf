import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from wolfplay.training.dpo import (
    DPOTrainingConfig,
    build_parser,
    config_from_args,
    train_dpo,
    validate_dpo_dataset,
)


def _write_dataset(path, **overrides):
    record = {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
    record.update(overrides)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _install_fake_training_modules(monkeypatch, captured):
    class FakeDataset:
        column_names = ["prompt", "chosen", "rejected", "metadata"]

        def __len__(self):
            return 1

        def remove_columns(self, columns):
            captured["removed_columns"] = columns
            self.column_names = [column for column in self.column_names if column not in columns]
            return self

    def load_dataset(*args, **kwargs):
        captured["load_dataset"] = (args, kwargs)
        return FakeDataset()

    class FakeDPOConfig:
        def __init__(self, **kwargs):
            captured["dpo_config"] = kwargs

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured["lora_config"] = kwargs

    class FakeDPOTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self):
            captured["trained"] = True

        def save_model(self, path):
            captured["saved_model"] = path

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    peft_module = ModuleType("peft")
    peft_module.LoraConfig = FakeLoraConfig
    peft_module.TaskType = SimpleNamespace(CAUSAL_LM="causal-lm")
    trl_module = ModuleType("trl")
    trl_module.DPOConfig = FakeDPOConfig
    trl_module.DPOTrainer = FakeDPOTrainer
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "trl", trl_module)


def test_validate_dpo_dataset_rejects_empty_and_bad_records(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no records"):
        validate_dpo_dataset(empty_path)

    missing_path = tmp_path / "missing.jsonl"
    missing_path.write_text(json.dumps({"prompt": "prompt"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_dpo_dataset(missing_path)

    identical_path = tmp_path / "identical.jsonl"
    _write_dataset(identical_path, chosen="same", rejected=" same ")
    with pytest.raises(ValueError, match="identical chosen and rejected"):
        validate_dpo_dataset(identical_path)


def test_training_config_validates_values_and_resolves_data_seed(tmp_path):
    config = DPOTrainingConfig(
        dataset=tmp_path / "dpo.jsonl",
        model="model",
        output_dir=tmp_path / "output",
        seed=7,
    )

    assert config.data_seed == 7
    assert config.lora_target_modules == "all-linear"
    with pytest.raises(ValueError, match="lora_dropout"):
        DPOTrainingConfig(
            dataset=tmp_path / "dpo.jsonl",
            model="model",
            output_dir=tmp_path / "output",
            lora_dropout=1.0,
        )
    with pytest.raises(ValueError, match="batch_size"):
        DPOTrainingConfig(
            dataset=tmp_path / "dpo.jsonl",
            model="model",
            output_dir=tmp_path / "output",
            batch_size=0,
        )


def test_parser_exposes_reproducibility_and_lora_options(tmp_path):
    args = build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "dpo.jsonl"),
            "--model",
            "model",
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            "11",
            "--data-seed",
            "12",
            "--lora-dropout",
            "0.1",
            "--lora-target-modules",
            "q_proj",
            "v_proj",
        ]
    )

    config = config_from_args(args)

    assert config.seed == 11
    assert config.data_seed == 12
    assert config.full_determinism is True
    assert config.lora_dropout == 0.1
    assert config.lora_target_modules == ("q_proj", "v_proj")


def test_train_dpo_wires_deterministic_config_lora_and_manifest(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dpo.jsonl"
    output_dir = tmp_path / "output"
    _write_dataset(dataset_path, metadata={"game_id": "game-1"})
    captured = {}
    _install_fake_training_modules(monkeypatch, captured)
    config = DPOTrainingConfig(
        dataset=dataset_path,
        model="test-model",
        output_dir=output_dir,
        seed=123,
        data_seed=456,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    train_dpo(config)

    assert captured["removed_columns"] == ["metadata"]
    assert captured["dpo_config"]["seed"] == 123
    assert captured["dpo_config"]["data_seed"] == 456
    assert captured["dpo_config"]["full_determinism"] is True
    assert captured["dpo_config"]["dataloader_num_workers"] == 0
    assert captured["dpo_config"]["precompute_ref_log_probs"] is False
    assert captured["lora_config"] == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "causal-lm",
        "inference_mode": False,
        "target_modules": "all-linear",
    }
    assert captured["trained"] is True
    assert captured["saved_model"] == str(output_dir / "final")

    manifest = json.loads((output_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == 123
    assert manifest["config"]["data_seed"] == 456
    assert manifest["dataset"]["records"] == 1
    assert len(manifest["dataset"]["sha256"]) == 64


def test_full_finetuning_precomputes_reference_log_probabilities(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dpo.jsonl"
    _write_dataset(dataset_path)
    captured = {}
    _install_fake_training_modules(monkeypatch, captured)

    train_dpo(
        DPOTrainingConfig(
            dataset=dataset_path,
            model="test-model",
            output_dir=tmp_path / "output",
            use_lora=False,
        )
    )

    assert captured["dpo_config"]["precompute_ref_log_probs"] is True
    assert "lora_config" not in captured
    assert captured["trainer"]["peft_config"] is None


def test_explicit_lora_target_modules_are_passed_as_a_list(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dpo.jsonl"
    _write_dataset(dataset_path)
    captured = {}
    _install_fake_training_modules(monkeypatch, captured)

    train_dpo(
        DPOTrainingConfig(
            dataset=dataset_path,
            model="test-model",
            output_dir=tmp_path / "output",
            lora_target_modules=("q_proj", "v_proj"),
        )
    )

    assert captured["lora_config"]["target_modules"] == ["q_proj", "v_proj"]
