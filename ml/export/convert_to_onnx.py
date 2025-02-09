"""Convert trained PyTorch model to ONNX format for fast inference."""

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def convert_to_onnx(
    model_dir: str = "ml/artifacts/complexity-classifier",
    output_path: str | None = None,
    opset_version: int = 14,
    optimize: bool = True,
) -> str:
    """
    Convert a trained HuggingFace model to ONNX format.

    Args:
        model_dir: Directory containing trained model
        output_path: Output path for ONNX model (defaults to model_dir/model.onnx)
        opset_version: ONNX opset version
        optimize: Whether to apply ONNX optimizations

    Returns:
        Path to the saved ONNX model
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path or model_dir / "model.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting model to ONNX")
    print(f"  Model dir: {model_dir}")
    print(f"  Output: {output_path}")
    print(f"  Opset: {opset_version}")

    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Create dummy input for tracing
    dummy_text = "This is a sample text for tracing the model."
    dummy_inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Define input names and dynamic axes
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size"},
    }

    # Export to ONNX
    print("\nExporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Model exported to: {output_path}")

    # Validate the model
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")

    # Apply optimizations if requested
    if optimize:
        print("\nApplying ONNX optimizations...")
        from onnxruntime.transformers import optimizer

        optimized_path = output_path.with_suffix(".optimized.onnx")
        optimized_model = optimizer.optimize_model(
            str(output_path),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        optimized_model.save_model_to_file(str(optimized_path))
        print(f"Optimized model saved to: {optimized_path}")

        # Use optimized model
        output_path = optimized_path

    # Verify inference
    print("\nVerifying inference...")
    _verify_onnx_inference(model, tokenizer, output_path)

    # Benchmark
    print("\nBenchmarking...")
    pytorch_time, onnx_time = _benchmark_inference(model, tokenizer, output_path)

    # Save conversion info
    info = {
        "original_model": str(model_dir),
        "onnx_path": str(output_path),
        "opset_version": opset_version,
        "optimized": optimize,
        "benchmark": {
            "pytorch_ms": pytorch_time,
            "onnx_ms": onnx_time,
            "speedup": pytorch_time / onnx_time if onnx_time > 0 else 0,
        },
    }

    info_path = output_path.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("=" * 50)
    print(f"\nONNX model: {output_path}")
    print(f"PyTorch latency: {pytorch_time:.2f}ms")
    print(f"ONNX latency: {onnx_time:.2f}ms")
    print(f"Speedup: {pytorch_time / onnx_time:.2f}x")

    return str(output_path)


def _verify_onnx_inference(model, tokenizer, onnx_path: Path) -> None:
    """Verify ONNX model produces same outputs as PyTorch."""
    # Test inputs
    test_texts = [
        "Hello, how are you?",
        "Write a Python function to calculate the factorial of a number recursively.",
    ]

    for text in test_texts:
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = model(**inputs)
            pytorch_logits = pytorch_outputs.logits.numpy()

        # ONNX inference
        session = ort.InferenceSession(str(onnx_path))
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        onnx_outputs = session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]

        # Compare
        np.testing.assert_allclose(pytorch_logits, onnx_logits, rtol=1e-3, atol=1e-4)

    print("  Inference verification passed!")


def _benchmark_inference(
    model, tokenizer, onnx_path: Path, num_runs: int = 100
) -> tuple[float, float]:
    """Benchmark PyTorch vs ONNX inference latency."""
    test_text = "What is the capital of France?"
    inputs = tokenizer(
        test_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Warmup
    with torch.no_grad():
        _ = model(**inputs)

    session = ort.InferenceSession(str(onnx_path))
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    _ = session.run(None, onnx_inputs)

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Benchmark ONNX
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = session.run(None, onnx_inputs)
    onnx_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    return pytorch_time, onnx_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert model to ONNX")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="ml/artifacts/complexity-classifier",
        help="Model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip ONNX optimizations",
    )

    args = parser.parse_args()

    convert_to_onnx(
        model_dir=args.model_dir,
        output_path=args.output,
        opset_version=args.opset,
        optimize=not args.no_optimize,
    )
