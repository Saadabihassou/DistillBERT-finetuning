from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Hugging Face model for SST-2 sentiment classification."
    )
    parser.add_argument(
        "--model-name",
        default="distilbert/distilbert-base-uncased",
        help="Hugging Face model ID (default: DistilBERT).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/distilbert-sst2",
        help="Directory where the fine-tuned model and tokenizer are saved.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--epochs", type=float, default=2, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional training subset size for quick experiments.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional validation subset size for quick experiments.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.epochs <= 0:
        parser.error("--epochs must be greater than 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    for name in ("max_train_samples", "max_eval_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    return args


def tokenize_function(tokenizer: Any, examples: dict[str, list[str]]) -> dict[str, Any]:
    """Tokenize a batch of SST-2 sentences, truncating overly long inputs."""
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute classification accuracy from Trainer's evaluation predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


def limit_split(dataset: Any, max_samples: int | None) -> Any:
    """Optionally select the first N examples to make experiments quicker."""
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    dataset = load_dataset("glue", "sst2")
    train_dataset = limit_split(dataset["train"], args.max_train_samples)
    eval_dataset = limit_split(dataset["validation"], args.max_eval_samples)

    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(tokenizer, examples), batched=True
    )
    eval_dataset = eval_dataset.map(
        lambda examples: tokenize_function(tokenizer, examples), batched=True
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Validation metrics: {metrics}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Fine-tuned model and tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    main()
