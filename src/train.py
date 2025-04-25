import argparse
import torch
from tqdm import tqdm
from data.dataset import get_dataloaders
from model.transformer_classifier import TransformerClassifier
from utils.utils import save_list
from evaluate import evaluate

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, valid_dl, test_dl, vocab, tokenizer = get_dataloaders(args.batch_size, device)
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_class=4,
        embedding_dim=args.embedding_dim,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    best_acc, losses, val_accs = 0.0, [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for labels, texts in tqdm(train_dl, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        val_acc = evaluate(valid_dl, model, device)
        val_accs.append(val_acc)
        print(f"Epoch {epoch} ▸ Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
        scheduler.step()
    save_list(losses, "loss.pkl")
    save_list(val_accs, "acc.pkl")
    print("Training complete. Best validation accuracy:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--lr",            type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int,   default=64)
    parser.add_argument("--nhead",         type=int,   default=5)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--num-layers",    type=int,   default=6)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--save-path",     type=str,   default="my_model.pth")
    args = parser.parse_args()
    train(args)