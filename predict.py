from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face text classifier.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="HF model name to use. Examples: distilbert-base-uncased, bert-base-uncased",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/distilbert-sst2",
        help="Directory to save model and tokenizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


def main() -> None:
    args = parse_args()

    # Basic reproducibility settings.
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    dataset = load_dataset("glue", "sst2")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
}
,{