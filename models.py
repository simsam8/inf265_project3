import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        out = F.relu(self.fc1(out))
        out = F.softmax(self.fc2(out))
