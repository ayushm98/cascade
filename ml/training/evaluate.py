"""Evaluate trained complexity classifier."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data.load_dataset import load_arc_dataset, load_easy2hard_bench


def evaluate_model(
    model_dir: str = "ml/artifacts/complexity-classifier",
    dataset_type: str = "arc",
    max_samples: int | None = None,
    output_dir: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Evaluate a trained complexity classifier.

    Args:
        model_dir: Directory containing trained model
        dataset_type: "easy2hard" or "arc"
        max_samples: Maximum samples to evaluate
        output_dir: Directory to save evaluation results (defaults to model_dir)
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir or model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating model from: {model_dir}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Load test data
    if dataset_type == "easy2hard":
        dataset = load_easy2hard_bench(max_samples=max_samples, seed=seed)
    else:
        dataset = load_arc_dataset(max_samples=max_samples, seed=seed)

    test_data = dataset["test"]
    print(f"Test set size: {len(test_data)}")

    # Run predictions
    all_labels = []
    all_predictions = []
    all_probabilities = []

    print("\nRunning predictions...")
    batch_size = 32

    for i in range(0, len(test_data), batch_size):
        batch = test_data.select(range(i, min(i + batch_size, len(test_data))))
        texts = batch["text"]
        labels = batch["label"]

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        all_labels.extend(labels)
        all_predictions.extend(preds.cpu().numpy().tolist())
        all_probabilities.extend(probs[:, 1].cpu().numpy().tolist())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(test_data))}/{len(test_data)}")

    # Convert to numpy
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    roc_auc = roc_auc_score(labels, probabilities)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    report = classification_report(
        labels,
        predictions,
        target_names=["simple", "complex"],
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    # Save results
    metrics = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels, predictions, target_names=["simple", "complex"], output_dict=True
        ),
    }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_dir / 'evaluation_metrics.json'}")

    # Generate plots
    _plot_confusion_matrix(cm, output_dir)
    _plot_roc_curve(labels, probabilities, output_dir)
    _plot_precision_recall_curve(labels, probabilities, output_dir)

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["simple", "complex"],
        yticklabels=["simple", "complex"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")


def _plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'roc_curve.png'}")


def _plot_precision_recall_curve(
    labels: np.ndarray, probabilities: np.ndarray, output_dir: Path
) -> None:
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'precision_recall_curve.png'}")


def analyze_errors(
    model_dir: str = "ml/artifacts/complexity-classifier",
    dataset_type: str = "arc",
    max_samples: int | None = None,
    num_examples: int = 10,
    seed: int = 42,
) -> None:
    """
    Analyze misclassified examples.

    Args:
        model_dir: Directory containing trained model
        dataset_type: "easy2hard" or "arc"
        max_samples: Maximum samples to evaluate
        num_examples: Number of error examples to show
        seed: Random seed
    """
    model_dir = Path(model_dir)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    if dataset_type == "easy2hard":
        dataset = load_easy2hard_bench(max_samples=max_samples, seed=seed)
    else:
        dataset = load_arc_dataset(max_samples=max_samples, seed=seed)

    test_data = dataset["test"]

    # Find errors
    false_positives = []  # Predicted complex, actually simple
    false_negatives = []  # Predicted simple, actually complex

    for example in test_data:
        text = example["text"]
        label = example["label"]

        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        if pred != label:
            error_info = {
                "text": text,
                "true_label": "complex" if label == 1 else "simple",
                "pred_label": "complex" if pred == 1 else "simple",
                "confidence": prob if pred == 1 else 1 - prob,
            }

            if pred == 1 and label == 0:
                false_positives.append(error_info)
            else:
                false_negatives.append(error_info)

    # Print analysis
    print("\n" + "=" * 60)
    print("Error Analysis")
    print("=" * 60)

    print(f"\nTotal errors: {len(false_positives) + len(false_negatives)}")
    print(f"  False positives (predicted complex, actually simple): {len(false_positives)}")
    print(f"  False negatives (predicted simple, actually complex): {len(false_negatives)}")

    print("\n--- False Positives (thought complex, was simple) ---")
    for i, error in enumerate(false_positives[:num_examples]):
        print(f"\n[{i+1}] Confidence: {error['confidence']:.2f}")
        print(f"    Text: {error['text'][:150]}...")

    print("\n--- False Negatives (thought simple, was complex) ---")
    for i, error in enumerate(false_negatives[:num_examples]):
        print(f"\n[{i+1}] Confidence: {error['confidence']:.2f}")
        print(f"    Text: {error['text'][:150]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate complexity classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="ml/artifacts/complexity-classifier",
        help="Model directory",
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
        default=None,
        help="Maximum samples",
    )
    parser.add_argument(
        "--analyze-errors",
        action="store_true",
        help="Show error analysis",
    )

    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        dataset_type=args.dataset,
        max_samples=args.max_samples,
    )

    if args.analyze_errors:
        analyze_errors(
            model_dir=args.model_dir,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
        )
