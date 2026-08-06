from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

_MAX_SEED = 2**32 - 1
_REQUIRED_COLUMNS = ("prompt", "chosen", "rejected")


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    dataset: Path
    model: str
    output_dir: Path
    epochs: float = 2.0
    learning_rate: float = 1e-6
    beta: float = 0.1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    seed: int = 42
    data_seed: int | None = None
    full_determinism: bool = True
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str | tuple[str, ...] = "all-linear"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", Path(self.dataset))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty model name or path")
        _require_positive_float("epochs", self.epochs)
        _require_positive_float("learning_rate", self.learning_rate)
        _require_positive_float("beta", self.beta)
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("gradient_accumulation_steps", self.gradient_accumulation_steps)
        _require_positive_int("max_length", self.max_length)
        _require_seed("seed", self.seed)
        data_seed = self.seed if self.data_seed is None else self.data_seed
        _require_seed("data_seed", data_seed)
        object.__setattr__(self, "data_seed", data_seed)
        if not isinstance(self.full_determinism, bool):
            raise ValueError("full_determinism must be a boolean")
        if not isinstance(self.use_lora, bool):
            raise ValueError("use_lora must be a boolean")
        _require_positive_int("lora_r", self.lora_r)
        _require_positive_int("lora_alpha", self.lora_alpha)
        if (
            isinstance(self.lora_dropout, bool)
            or not isinstance(self.lora_dropout, (int, float))
            or not math.isfinite(self.lora_dropout)
            or not 0 <= self.lora_dropout < 1
        ):
            raise ValueError("lora_dropout must be a finite number in [0, 1)")
        object.__setattr__(
            self,
            "lora_target_modules",
            _normalize_target_modules(self.lora_target_modules),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": str(self.dataset.resolve()),
            "model": self.model,
            "output_dir": str(self.output_dir.resolve()),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "beta": self.beta,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "full_determinism": self.full_determinism,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
        }


@dataclass(frozen=True, slots=True)
class DPODataSummary:
    records: int
    sha256: str


def validate_dpo_dataset(path: Path) -> DPODataSummary:
    """Validate a standard explicit-prompt DPO JSONL file without loading ML packages."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"DPO dataset does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"DPO dataset is not a file: {path}")

    digest = hashlib.sha256()
    record_count = 0
    with path.open("rb") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"DPO dataset must be UTF-8 encoded: {path}, line {line_number}"
                ) from error
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid DPO JSONL at {path}: line {line_number}, column {error.colno}: "
                    f"{error.msg}"
                ) from error
            _validate_dpo_record(record, path=path, line_number=line_number)
            record_count += 1

    if record_count == 0:
        raise ValueError(f"DPO dataset contains no records: {path}")
    return DPODataSummary(records=record_count, sha256=digest.hexdigest())


def train_dpo(config: DPOTrainingConfig) -> None:
    data_summary = validate_dpo_dataset(config.dataset)
    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from trl import DPOConfig, DPOTrainer
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are missing; run `uv sync --extra train` first"
        ) from error

    try:
        dataset = load_dataset("json", data_files=str(config.dataset), split="train")
    except Exception as error:
        raise ValueError(
            f"failed to load validated DPO dataset {config.dataset}: {error}"
        ) from error
    missing = [column for column in _REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"DPO dataset is missing columns: {missing}")
    removable = sorted(column for column in dataset.column_names if column not in _REQUIRED_COLUMNS)
    if removable:
        dataset = dataset.remove_columns(removable)
    if len(dataset) == 0:
        raise ValueError(f"DPO dataset contains no records after loading: {config.dataset}")

    training_args = DPOConfig(
        output_dir=str(config.output_dir),
        beta=config.beta,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        seed=config.seed,
        data_seed=config.data_seed,
        full_determinism=config.full_determinism,
        dataloader_num_workers=0,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        precompute_ref_log_probs=not config.use_lora,
    )
    peft_config = None
    if config.use_lora:
        target_modules = config.lora_target_modules
        if isinstance(target_modules, tuple):
            target_modules = list(target_modules)
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_training_manifest(config, data_summary)
    trainer = DPOTrainer(
        model=config.model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir / "final"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a WolfPlay policy with TRL DPOTrainer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument(
        "--no-full-determinism",
        action="store_false",
        dest="full_determinism",
        help="disable deterministic algorithms for higher throughput",
    )
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["all-linear"])
    return parser


def config_from_args(args: argparse.Namespace) -> DPOTrainingConfig:
    target_modules: str | tuple[str, ...]
    if len(args.lora_target_modules) == 1:
        target_modules = args.lora_target_modules[0]
    else:
        target_modules = tuple(args.lora_target_modules)
    return DPOTrainingConfig(
        dataset=args.dataset,
        model=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        seed=args.seed,
        data_seed=args.data_seed,
        full_determinism=args.full_determinism,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
    )


def _validate_dpo_record(record: object, *, path: Path, line_number: int) -> None:
    context = f"{path}: line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = [column for column in _REQUIRED_COLUMNS if column not in record]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    for column in _REQUIRED_COLUMNS:
        value = record[column]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context}.{column} must be a non-empty string")
    if record["chosen"].strip() == record["rejected"].strip():
        raise ValueError(f"{context} has identical chosen and rejected responses")


def _write_training_manifest(config: DPOTrainingConfig, data_summary: DPODataSummary) -> None:
    manifest = {
        "config": config.to_dict(),
        "dataset": {
            "path": str(config.dataset.resolve()),
            "records": data_summary.records,
            "sha256": data_summary.sha256,
        },
        "packages": {
            package: _package_version(package)
            for package in ("datasets", "peft", "transformers", "trl")
        },
    }
    manifest_path = config.output_dir / "training_manifest.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=config.output_dir,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _require_positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_seed(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_SEED:
        raise ValueError(f"{name} must be an integer in [0, {_MAX_SEED}]")


def _normalize_target_modules(value: str | tuple[str, ...]) -> str | tuple[str, ...]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("lora_target_modules must not be empty")
        return value.strip()
    try:
        modules = tuple(value)
    except TypeError as error:
        raise ValueError("lora_target_modules must be a string or sequence of strings") from error
    if not modules or any(not isinstance(module, str) or not module.strip() for module in modules):
        raise ValueError("lora_target_modules must contain non-empty strings")
    return tuple(module.strip() for module in modules)


def main() -> None:
    args = build_parser().parse_args()
    train_dpo(config_from_args(args))


if __name__ == "__main__":
    main()
