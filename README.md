# Transformer-Based Text Classifier

## Overview
This project implements a transformer‐encoder text classifier on the AG_NEWS dataset using PyTorch and TorchText. The code is organized into modules for data, model, training, evaluation, visualization and utilities.

## Installation
1. Clone the repo  
2. Create a virtual env and activate it  
3. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training
```
python -m src.train \
  --batch-size 64 \
  --epochs 10 \
  --lr 0.1 \
  --save-path my_model.pth
```

### Evaluation
```
python -m src.evaluate \
  --model-path my_model.pth
```

### Inference
```python
from src.evaluate import predict, load_model
model, vocab, tokenizer, device = load_model("my_model.pth")
print(predict("This is a test sentence", model, vocab, tokenizer, device))
```

## Project Structure
- **src/data/**: data pipelines & dataloaders  
- **src/model/**: transformer classifier implementation  
- **src/utils/**: saving/loading utilities  
- **src/visualization.py**: plotting functions  
- **src/train.py**: training script  
- **src/evaluate.py**: evaluation & inference