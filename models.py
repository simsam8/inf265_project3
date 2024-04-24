import torch
import math
import numpy as np
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
    def __init__(self, embedding, max_len):
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
        self.fc1 = nn.Linear(embedding_dim * max_len, 128)
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
    def __init__(self, embedding, max_len, n_heads=4, w_size=8) -> None:
        super().__init__()

        self.max_len = max_len
        (vocab_size, embedding_dim) = embedding.weight.shape
        # Instantiate an embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Load the pretrained weights
        self.embedding.load_state_dict(embedding.state_dict())
        # Freeze the layer
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.positional_encoding = PositionalEncoding(max_len, embedding_dim)
        self.multi_head = MultiHeadLayer(n_heads, embedding_dim, w_size)
        self.fc1 = nn.Linear(max_len * embedding_dim, 12)

    def forward(self, x):
        out = self.embedding(x)
        out = self.positional_encoding(out)
        out = self.multi_head(out)
        out = self.fc1(torch.flatten(out, 1))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, embedding_dim, n=10000) -> None:
        super().__init__()
        # calculate the div_term
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * -(math.log(n) / embedding_dim)
        )

        # generate the positions into a column matrix
        k = torch.arange(0, max_length).unsqueeze(1)

        # generate an empty tensor
        pe = torch.zeros(max_length, embedding_dim)

        # set the even values
        pe[:, 0::2] = torch.sin(k * div_term)

        # set the odd values
        pe[:, 1::2] = torch.cos(k * div_term)

        # add a dimension
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # the output has a shape of (1, max_leng, embeding_dim)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class MultiHeadLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim, w_size) -> None:
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [SingleHead(embedding_dim, w_size) for _ in range(n_heads)]
        )
        self.attention_out = nn.Linear(w_size * n_heads, embedding_dim)

    def forward(self, x):
        z_outputs = [head(x) for head in self.attention_heads]
        out = torch.cat(z_outputs, dim=2)
        out = self.attention_out(out)
        return out


class SingleHead(nn.Module):
    def __init__(self, embedding_dim, w_size) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_dim, w_size)
        self.key = nn.Linear(embedding_dim, w_size)
        self.value = nn.Linear(embedding_dim, w_size)
        self.scaled_dot_product_attention = ScaledDotProductAttention(w_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        out = self.scaled_dot_product_attention(query, key, value)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim) -> None:
        """
        Use the same dimension for query, key, and value matrices
        """
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attention = F.softmax(score, -1)
        context = torch.bmm(attention, value)
        return context  # , attention


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
        self.rnn = nn.RNN(
            num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(
            num_hiddens, 12
        )  # Between recurrent and output. There are 12 possible conjugations

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out = self.rnn(out, hidden)
        out = out[:, -1, :]  # Get the last time step's output
        out = self.fc(out)  # Fully connected output layer
        return out


class GenerativeRNN(nn.Module):
    def __init__(self, embedding, num_inputs, num_hiddens, num_layers, dropout=0):
        super().__init__()

        # Embedding layer
        (vocab_size, embedding_dim) = embedding.weight.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict(embedding.state_dict())
        for p in self.embedding.parameters():
            p.requires_grad = False

        # Recurrent layer(s)
        self.rnn = nn.RNN(
            num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(num_hiddens, vocab_size)  # Between recurrent and output

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out = self.rnn(out, hidden)
        out = out[:, -1, :]  # Get the last time step's output
        out = self.fc(out)  # Fully connected output layer
        return out


class GenerativeLSTM(nn.Module):
    def __init__(
        self, embedding, num_inputs, num_hiddens, num_layers, dropout=0
    ):  # Use dropout if num_layers > 1
        super().__init__()

        # Embedding layer
        (vocab_size, embedding_dim) = embedding.weight.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.load_state_dict(embedding.state_dict())
        for p in self.embedding.parameters():
            p.requires_grad = False

        # LSTM layer(s)
        self.lstm = nn.LSTM(
            num_inputs, num_hiddens, num_layers, dropout=dropout, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(num_hiddens, vocab_size)  # Between recurrent and output

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.lstm(out, hidden)
        out = out[:, -1, :]  # Get the last time step's output
        out = self.fc(out)  # Fully connected output layer
        return out, hidden
