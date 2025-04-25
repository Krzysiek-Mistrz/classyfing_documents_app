import torch
import argparse

from data.dataset import get_dataloaders
from model.transformer_classifier import TransformerClassifier

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tec"}

def evaluate(dataloader, model, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for labels, texts in dataloader:
            outputs = model(texts.to(device))
            preds = outputs.argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)
    return correct / total

def load_model(model_path, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_dl, vocab, tokenizer = get_dataloaders(batch_size, device)
    model = TransformerClassifier(vocab_size=len(vocab), num_class=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, vocab, tokenizer, device

def predict(text, model, vocab, tokenizer, device):
    token_ids = torch.tensor(vocab(tokenizer(text)), dtype=torch.int64).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(token_ids)
        pred = output.argmax(dim=1).item()
    return AG_NEWS_LABELS[pred]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    model, vocab, tokenizer, device = load_model(args.model_path, args.batch_size)
    _, _, test_dl, _, _ = get_dataloaders(args.batch_size, device)
    test_acc = evaluate(test_dl, model, device)
    print(f"Test Accuracy: {test_acc:.4f}")