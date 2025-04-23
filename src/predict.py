import argparse
import torch
from torchtext.data.utils import get_tokenizer
from data.dataset import build_vocab, text_pipeline
from model.text_classification import TextClassificationModel

AG_NEWS_LABELS = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tec'}

def predict(model_path, sentence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab(tokenizer)
    model = TextClassificationModel(len(vocab), embed_dim=100, num_class=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        text_ids = torch.tensor(text_pipeline(sentence, vocab, tokenizer), dtype=torch.int64).to(device)
        output = model(text_ids, torch.tensor([0], device=device))
        label_idx = output.argmax(1).item() + 1
        return AG_NEWS_LABELS[label_idx]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Predict single AG_NEWS label")
    p.add_argument('--model-path', type=str, default='model.pth')
    p.add_argument('--sentence',   type=str, required=True)
    args = p.parse_args()
    prediction = predict(args.model_path, args.sentence)
    print(f"Predicted label: {prediction}")