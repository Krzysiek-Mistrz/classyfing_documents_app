import argparse
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from data.dataset import get_dataloaders
from model.text_classification import TextClassificationModel

AG_NEWS_LABELS = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tec'}

def evaluate(model, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for labels, texts, offsets in dataloader:
            outputs = model(texts, offsets)
            total_acc += (outputs.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab, tokenizer, train_loader, valid_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, valid_ratio=args.valid_ratio, device=device
    )
    model = TextClassificationModel(len(vocab), args.embed_dim, len(AG_NEWS_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_acc = 0.0
    train_losses, val_accs = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for labels, texts, offsets in train_loader:
            optimizer.zero_grad()
            outputs = model(texts, offsets)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        val_acc = evaluate(model, valid_loader, device)
        train_losses.append(epoch_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch}/{args.epochs} — Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
    test_acc = evaluate(model, test_loader, device)
    print(f"Best Val Acc: {best_acc:.4f}, Test Acc: {test_acc:.4f}")
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs+1), val_accs,    label='Val Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_plot.png')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train text classifier on AG_NEWS")
    p.add_argument('--epochs',      type=int,   default=10)
    p.add_argument('--batch-size',  dest='batch_size', type=int, default=64)
    p.add_argument('--lr',          dest='lr',         type=float, default=0.1)
    p.add_argument('--gamma',       type=float, default=0.1)
    p.add_argument('--embed-dim',   dest='embed_dim', type=int,   default=100)
    p.add_argument('--valid-ratio', dest='valid_ratio', type=float, default=0.05)
    p.add_argument('--save-path',   dest='save_path', type=str, default='model.pth')
    args = p.parse_args()
    train(args)