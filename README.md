# AG_NEWS Text Classification

A simple PyTorch/TorchText project for classifying news articles from the AG_NEWS dataset using an `EmbeddingBag + Linear` model.

## Project Structure

```
.
├── src/
│   ├── data/
│   │   └── dataset.py        # Vocab building & DataLoader utils
│   ├── model/
│   │   └── text_classification.py  # Model definition
│   ├── train.py              # Training & evaluation script
│   └── predict.py            # Single‐sentence prediction script
├── requirements.txt
└── README.md
```

## Installation

1. Clone repo:
   ```bash
   git clone <your-repo-url>
   cd <repo-root>
   ```
2. (Optional) create virtual env:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the training script with defaults:
```bash
python -m src.train \
  --epochs 10 \
  --batch-size 64 \
  --lr 0.1 \
  --embed-dim 100 \
  --valid-ratio 0.05 \
  --save-path model.pth
```
- Outputs `model.pth` (best validation) and `training_plot.png`.

## Prediction

After training, feed any sentence for a label prediction:
```bash
python -m src.predict \
  --model-path model.pth \
  --sentence "Intergalactic stocks soar after new tech release"
```

## License

GNU GPL V3 © Krzychu