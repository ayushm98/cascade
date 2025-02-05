"""Train DistilBERT for complexity classification."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.load_dataset import load_arc_dataset, load_easy2hard_bench


def compute_metrics(eval_pred) -> dict:
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
    }


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize the dataset."""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,  # Will be handled by data collator
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "difficulty_score"],
        desc="Tokenizing",
    )

    return tokenized


def train_complexity_classifier(
    model_name: str = "distilbert-base-uncased",
    dataset_type: str = "arc",
    max_samples: int | None = 5000,
    output_dir: str = "ml/artifacts/complexity-classifier",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
) -> dict:
    """
    Train a DistilBERT model for complexity classification.

    Args:
        model_name: HuggingFace model name
        dataset_type: "easy2hard" or "arc"
        max_samples: Maximum training samples (None for all)
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        seed: Random seed

    Returns:
        Dictionary with training metrics
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training complexity classifier")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_type}")
    print(f"  Output: {output_dir}")
    print()

    # Load dataset
    if dataset_type == "easy2hard":
        dataset = load_easy2hard_bench(max_samples=max_samples, seed=seed)
    else:
        dataset = load_arc_dataset(max_samples=max_samples, seed=seed)

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "simple", 1: "complex"},
        label2id={"simple": 0, "complex": 1},
    )

    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])

    # Save the model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metrics
    metrics = {
        "train": {
            "loss": train_result.training_loss,
            "epochs": train_result.metrics.get("epoch", num_epochs),
        },
        "test": {
            "accuracy": test_metrics["eval_accuracy"],
            "f1": test_metrics["eval_f1"],
            "precision": test_metrics["eval_precision"],
            "recall": test_metrics["eval_recall"],
            "loss": test_metrics["eval_loss"],
        },
        "config": {
            "model_name": model_name,
            "dataset_type": dataset_type,
            "max_samples": max_samples,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['eval_accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['eval_f1']:.4f}")
    print(f"  Precision: {test_metrics['eval_precision']:.4f}")
    print(f"  Recall:    {test_metrics['eval_recall']:.4f}")
    print(f"\nModel saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train complexity classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dataset",
        choices=["easy2hard", "arc"],
        default="arc",
        help="Dataset to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum samples (None for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/artifacts/complexity-classifier",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    train_complexity_classifier(
        model_name=args.model,
        dataset_type=args.dataset,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
