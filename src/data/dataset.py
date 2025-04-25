import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torch.nn.utils.rnn import pad_sequence

def yield_tokens(data_iter, tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def build_vocab(tokenizer, specials=["<unk>"]):
    train_iter = AG_NEWS(split="train")
    vocab = build_vocab_from_iterator(yield_tokens(train_iter, tokenizer), specials=specials)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def text_pipeline(text, vocab, tokenizer):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return int(label) - 1

def collate_batch(batch, vocab, tokenizer, device):
    labels, texts = [], []
    for lbl, txt in batch:
        labels.append(label_pipeline(lbl))
        tokens = torch.tensor(text_pipeline(txt, vocab, tokenizer), dtype=torch.int64)
        texts.append(tokens)
    labels = torch.tensor(labels, dtype=torch.int64)
    texts = pad_sequence(texts, batch_first=True)
    return labels.to(device), texts.to(device)

def get_dataloaders(batch_size, device):
    """
    Returns: train_loader, valid_loader, test_loader, vocab, tokenizer
    """
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(tokenizer)
    train_iter, test_iter = AG_NEWS()
    train_ds = to_map_style_dataset(train_iter)
    test_ds  = to_map_style_dataset(test_iter)
    num_train = int(len(train_ds) * 0.95)
    train_ds, valid_ds = random_split(train_ds, [num_train, len(train_ds) - num_train])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "collate_fn": lambda b: collate_batch(b, vocab, tokenizer, device),
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  **loader_kwargs)
    return train_loader, valid_loader, test_loader, vocab, tokenizer