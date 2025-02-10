"""ONNX-based complexity classifier for fast inference."""

import asyncio
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class ONNXComplexityClassifier:
    """
    Complexity classifier using ONNX Runtime for fast inference.

    This classifier determines whether a query is "simple" or "complex"
    to enable intelligent routing to appropriate LLM models.

    Attributes:
        model_path: Path to the ONNX model file
        tokenizer_path: Path to the tokenizer directory
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        model_path: str | Path = "models/model.onnx",
        tokenizer_path: str | Path | None = None,
        max_length: int = 128,
    ) -> None:
        """
        Initialize the complexity classifier.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer (defaults to model directory)
            max_length: Maximum sequence length

        Raises:
            ImportError: If onnxruntime or transformers not installed
            FileNotFoundError: If model file doesn't exist
        """
        if ort is None:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        if AutoTokenizer is None:
            raise ImportError("transformers is required. Install with: pip install transformers")

        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path or self.model_path.parent)
        self.max_length = max_length

        self._session: ort.InferenceSession | None = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load ONNX model with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CPU for inference (can be extended for GPU)
        providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            self.load()

    def classify_sync(self, text: str) -> float:
        """
        Classify text complexity synchronously.

        Args:
            text: Input text to classify

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        self._ensure_loaded()

        # Tokenize
        inputs = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Run inference
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        outputs = self._session.run(None, onnx_inputs)
        logits = outputs[0][0]

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Return probability of "complex" class (index 1)
        return float(probs[1])

    async def classify(self, text: str) -> float:
        """
        Classify text complexity asynchronously.

        Uses thread pool to avoid blocking the event loop since
        ONNX inference is CPU-bound.

        Args:
            text: Input text to classify

        Returns:
            Complexity score between 0.0 (simple) and 1.0 (complex)
        """
        return await asyncio.to_thread(self.classify_sync, text)

    def classify_batch_sync(self, texts: list[str]) -> list[float]:
        """
        Classify multiple texts synchronously.

        Args:
            texts: List of texts to classify

        Returns:
            List of complexity scores
        """
        self._ensure_loaded()

        # Tokenize batch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        # Run inference
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        outputs = self._session.run(None, onnx_inputs)
        logits = outputs[0]

        # Softmax for each sample
        scores = []
        for sample_logits in logits:
            exp_logits = np.exp(sample_logits - np.max(sample_logits))
            probs = exp_logits / exp_logits.sum()
            scores.append(float(probs[1]))

        return scores

    async def classify_batch(self, texts: list[str]) -> list[float]:
        """
        Classify multiple texts asynchronously.

        Args:
            texts: List of texts to classify

        Returns:
            List of complexity scores
        """
        return await asyncio.to_thread(self.classify_batch_sync, texts)

    def get_label(self, score: float) -> Literal["simple", "complex"]:
        """
        Convert score to label.

        Args:
            score: Complexity score

        Returns:
            "simple" if score < 0.5, else "complex"
        """
        return "complex" if score >= 0.5 else "simple"

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"ONNXComplexityClassifier(model={self.model_path}, {status})"


# Convenience function for quick classification
def classify_complexity(text: str, model_path: str = "models/model.onnx") -> float:
    """
    Quick function to classify text complexity.

    Args:
        text: Text to classify
        model_path: Path to ONNX model

    Returns:
        Complexity score (0.0 = simple, 1.0 = complex)
    """
    classifier = ONNXComplexityClassifier(model_path)
    return classifier.classify_sync(text)
