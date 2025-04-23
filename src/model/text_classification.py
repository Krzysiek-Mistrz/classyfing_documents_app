import torch
from torch import nn

class TextClassificationModel(nn.Module):
    """
    A simple EmbeddingBag + Linear model for text classification.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self._init_weights()
    def _init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)