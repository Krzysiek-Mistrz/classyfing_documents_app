import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

def build_vocab(tokenizer):
    """
    Build a vocab from the AG_NEWS training split.
    """
    train_iter = AG_NEWS(split='train')
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text.lower())
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def text_pipeline(text, vocab, tokenizer):
    """
    Tokenize & turn a raw string into a list of token IDs.
    """
    return vocab(tokenizer(text.lower()))

def label_pipeline(label):
    """
    Convert 1-based label to 0-based integer.
    """
    return int(label) - 1

def collate_fn(batch, vocab, tokenizer, device):
    """
    Collate a batch of (label,text) pairs into tensors.
    """
    label_list, text_list, offsets = [], [], [0]
    for label, text in batch:
        label_list.append(label_pipeline(label))
        processed = torch.tensor(text_pipeline(text, vocab, tokenizer), dtype=torch.int64)
        text_list.append(processed)
        offsets.append(processed.size(0))
    label_tensor = torch.tensor(label_list, dtype=torch.int64)
    offsets_tensor = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_tensor = torch.cat(text_list)
    return (label_tensor.to(device),
            text_tensor.to(device),
            offsets_tensor.to(device))

def get_dataloaders(batch_size=64, valid_ratio=0.05, device=torch.device('cpu')):
    """
    Returns: vocab, tokenizer, train_loader, valid_loader, test_loader
    """
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab(tokenizer)
    train_iter, test_iter = AG_NEWS()
    train_ds = to_map_style_dataset(train_iter)
    test_ds  = to_map_style_dataset(test_iter)
    num_train = int(len(train_ds) * (1 - valid_ratio))
    split_train, split_valid = random_split(train_ds, [num_train, len(train_ds)-num_train])
    train_loader = DataLoader(split_train, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, vocab, tokenizer, device))
    valid_loader = DataLoader(split_valid, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, vocab, tokenizer, device))
    test_loader  = DataLoader(test_ds,    batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, vocab, tokenizer, device))
    return vocab, tokenizer, train_loader, valid_loader, test_loader