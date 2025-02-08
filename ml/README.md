# ML Pipeline

This directory contains the machine learning pipeline for training the complexity classifier.

## Structure

```
ml/
├── data/              # Dataset loading and preprocessing
│   └── load_dataset.py
├── training/          # Model training and evaluation
│   ├── train.py       # DistilBERT fine-tuning
│   └── evaluate.py    # Model evaluation
├── export/            # Model export
│   └── convert_to_onnx.py
└── artifacts/         # Saved models and metrics
    ├── model.onnx
    └── metrics.json
```

## Training

```bash
# Train the complexity classifier
python -m ml.training.train --dataset arc --epochs 5

# Evaluate the model
python -m ml.training.evaluate --model-dir ml/artifacts/complexity-classifier

# Export to ONNX
python -m ml.export.convert_to_onnx --model-dir ml/artifacts/complexity-classifier
```

## Dataset

The classifier is trained on the ARC dataset (AI2 Reasoning Challenge) which provides:
- **Easy examples**: Simple questions that can be handled by smaller models
- **Challenge examples**: Complex questions requiring more capable models

Alternatively, Easy2Hard-Bench can be used for continuous difficulty scores.
