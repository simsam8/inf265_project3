import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim=16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * context_size * 2, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        out = out.flatten(1, 2)
        out = F.relu(self.fc1(out))
        out = F.log_softmax(self.fc2(out), dim=1)
        return out


class SimpleMLP(nn.Module):
    def __init__(self, embedding, context_size):
        super().__init__()

        (vocab_size, embedding_dim) = embedding.weight.shape
        # Instantiate an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Load the pretrained weights
        self.embedding.load_state_dict(embedding.state_dict())
        # Freeze the layer
        for p in self.embedding.parameters():
            p.requires_grad = False

        # Regular MLP
        self.fc1 = nn.Linear(embedding_dim * context_size, 128)
        self.fc2 = nn.Linear(128, 12)

    def forward(self, x):
        # x is of shape (N, context_size) but contains integers which can
        # be seen as equivalent to (N, context_size, vocab_size) since one hot
        # encoding is used under the hood
        out = self.embedding(x)
        # out is now of shape (N, context_size, embedding_dim)

        out = F.relu(self.fc1(torch.flatten(out, 1)))
        # out is now of shape (N, context_size*embedding_dim)

        out = self.fc2(out)
        return out


class AttentionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass


class ConjugationRNN(nn.Module):
    def __init__(self, embedding, num_inputs, num_hiddens, num_layers, dropout=0):
        super().__init__()

        # Embedding layer with pretrained weights
        (vocab_size, embedding_dim) = embedding.weight.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict(embedding.state_dict())
        # Freeze the embedding layer to avoid weight updates during training
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Recurrent layer(s)
        self.rnn = nn.RNN(num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(num_hiddens, 12)  # Between recurrent and output. There are 12 possible conjugations

    def forward(self, x, H=None):
        out = self.embedding(x)
        out = self.rnn(out, H)
        out = out[:, -1, :]  # Get the last time step's output
        out = self.fc(out)  # Fully connected output layer
        return out

"""
Use your trained word embedding and define a RNN architecture that can predict the next
word given the context before the target.
"""
class GenerationRNN(nn.Module):
    def __init__(self, embedding, num_inputs, num_hiddens, num_layers, dropout=0):
        super().__init__()

                # Embedding layer with pretrained weights
        (vocab_size, embedding_dim) = embedding.weight.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict(embedding.state_dict())
        # Freeze the embedding layer to avoid weight updates during training
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        # Recurrent layer(s)
        self.rnn = nn.RNN(num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(num_hiddens, vocab_size)  # Between recurrent and output
