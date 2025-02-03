"""Load and preprocess Easy2Hard-Bench dataset for complexity classification."""

import json
from pathlib import Path
from typing import Literal

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def load_easy2hard_bench(
    subset: Literal["all", "gsm8k", "arc", "winogrande"] = "all",
    difficulty_threshold: float = 0.5,
    max_samples: int | None = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Load Easy2Hard-Bench dataset and convert to binary classification.

    Args:
        subset: Which subset to load ("all" for combined dataset)
        difficulty_threshold: Score above this is "complex" (1), below is "simple" (0)
        max_samples: Maximum samples to use (None for all)
        seed: Random seed for shuffling

    Returns:
        DatasetDict with train/validation/test splits
    """
    print(f"Loading Easy2Hard-Bench dataset (subset={subset})...")

    # Load the dataset from HuggingFace
    dataset = load_dataset("furonghuang-lab/Easy2Hard-Bench")

    # Get all available splits
    all_data = []

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        all_data.append(split_data)

    # Combine all splits
    combined = concatenate_datasets(all_data)

    print(f"Total examples loaded: {len(combined)}")

    # Process the dataset
    def process_example(example: dict) -> dict:
        """Extract text and create binary label from difficulty score."""
        # Get the question/prompt text
        text = example.get("question", "") or example.get("prompt", "") or example.get("input", "")

        # Get difficulty score (normalize to 0-1 if needed)
        difficulty = example.get("difficulty", 0.5)

        # Convert to binary label
        label = 1 if difficulty >= difficulty_threshold else 0

        return {
            "text": str(text).strip(),
            "label": label,
            "difficulty_score": float(difficulty),
        }

    # Apply processing
    processed = combined.map(
        process_example,
        remove_columns=combined.column_names,
        desc="Processing examples",
    )

    # Filter out empty texts
    processed = processed.filter(lambda x: len(x["text"]) > 0)

    print(f"After filtering empty texts: {len(processed)}")

    # Shuffle the dataset
    processed = processed.shuffle(seed=seed)

    # Limit samples if specified
    if max_samples and len(processed) > max_samples:
        processed = processed.select(range(max_samples))
        print(f"Limited to {max_samples} samples")

    # Create train/val/test splits (70/15/15)
    train_test = processed.train_test_split(test_size=0.3, seed=seed)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    # Print statistics
    print("\nDataset splits:")
    for split_name, split_data in dataset_dict.items():
        n_simple = sum(1 for x in split_data if x["label"] == 0)
        n_complex = sum(1 for x in split_data if x["label"] == 1)
        print(f"  {split_name}: {len(split_data)} total ({n_simple} simple, {n_complex} complex)")

    return dataset_dict


def load_arc_dataset(max_samples: int | None = None, seed: int = 42) -> DatasetDict:
    """
    Load ARC dataset with pre-defined Easy/Challenge splits.

    This is an alternative to Easy2Hard-Bench that has explicit easy/hard labels.

    Args:
        max_samples: Maximum samples per split (None for all)
        seed: Random seed for shuffling

    Returns:
        DatasetDict with train/validation/test splits
    """
    print("Loading ARC dataset (Easy + Challenge)...")

    # Load both splits
    arc_easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def process_arc(example: dict, label: int) -> dict:
        """Process ARC example."""
        return {
            "text": example["question"].strip(),
            "label": label,
            "difficulty_score": 0.25 if label == 0 else 0.75,
        }

    # Process and label
    easy_data = arc_easy["train"].map(
        lambda x: process_arc(x, 0),
        remove_columns=arc_easy["train"].column_names,
    )
    challenge_data = arc_challenge["train"].map(
        lambda x: process_arc(x, 1),
        remove_columns=arc_challenge["train"].column_names,
    )

    # Combine
    combined = concatenate_datasets([easy_data, challenge_data])
    combined = combined.shuffle(seed=seed)

    if max_samples and len(combined) > max_samples:
        combined = combined.select(range(max_samples))

    # Split
    train_test = combined.train_test_split(test_size=0.3, seed=seed)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=seed)

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    print("\nDataset splits:")
    for split_name, split_data in dataset_dict.items():
        n_simple = sum(1 for x in split_data if x["label"] == 0)
        n_complex = sum(1 for x in split_data if x["label"] == 1)
        print(f"  {split_name}: {len(split_data)} total ({n_simple} simple, {n_complex} complex)")

    return dataset_dict


def save_dataset(dataset: DatasetDict, output_dir: str | Path) -> None:
    """Save processed dataset to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for example in split_data:
                f.write(json.dumps(example) + "\n")
        print(f"Saved {split_name} to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Load complexity classification dataset")
    parser.add_argument(
        "--dataset",
        choices=["easy2hard", "arc"],
        default="easy2hard",
        help="Dataset to load",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Difficulty threshold for binary classification",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed dataset to disk",
    )

    args = parser.parse_args()

    if args.dataset == "easy2hard":
        dataset = load_easy2hard_bench(
            difficulty_threshold=args.threshold,
            max_samples=args.max_samples,
        )
    else:
        dataset = load_arc_dataset(max_samples=args.max_samples)

    if args.save:
        save_dataset(dataset, args.output_dir)

    # Show some examples
    print("\nSample examples:")
    for i, example in enumerate(dataset["train"].select(range(3))):
        label_str = "complex" if example["label"] == 1 else "simple"
        print(f"\n[{i+1}] ({label_str}, score={example['difficulty_score']:.2f})")
        print(f"    {example['text'][:100]}...")
