import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from models import GenerativeLSTM, GenerativeRNN

def set_device(device=None):
    """
    Helper function to set device
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"On device {device}.")
    return device


def compute_accuracy(model: nn.Module, loader: DataLoader, device=None):

    correct, total = 0, 0

    model.eval()
    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            log_probs = model(contexts)
            predictions = torch.argmax(log_probs, dim=1)

            total += targets.shape[0]
            correct += int((predictions == targets).sum())

    # Calculate accuracy
    acc = correct / total

    return acc


def train(epochs, model, optimizer, loss_fn, train_loader, val_loader, device=None):
    n_batch_train = len(train_loader)
    train_losses = []
    train_accuracies = []

    n_batch_val = len(val_loader)
    val_losses = []
    val_accuracies = []

    optimizer.zero_grad()
    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        model.train()
        for contexts, targets in train_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            log_probs = model(contexts)

            loss = loss_fn(log_probs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        train_losses.append(train_loss / n_batch_train)

        train_acc = compute_accuracy(model, train_loader, device)
        train_accuracies.append(train_acc)

        model.eval()
        with torch.no_grad():
            for contexts, targets in val_loader:
                contexts = contexts.to(device)
                targets = targets.to(device)
                log_probs = model(contexts)

                loss = loss_fn(log_probs, targets)
                val_loss += loss.item()

            val_losses.append(val_loss / n_batch_val)

            val_acc = compute_accuracy(model, val_loader, device)
            val_accuracies.append(val_acc)

        if epoch == 1 or epoch % 1 == 0:
            log = (
                f"{datetime.now().time()}, Epoch: {epoch}, "
                + f"train_loss: {train_loss/n_batch_train:.3f}, train_accuracy: {train_acc*100:.3f}%, "
                + f"val_loss: {val_loss/n_batch_val:.3f}, val_accuracy: {val_acc*100:.3f}%"
            )
            print(log)
    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_performance_over_time(
    train_perf: list[float],
    val_perf: list[float],
    title: str,
    y_label: str,
) -> None:
    """
    Creates a plot of training and validation loss/performance over time.
    """
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(train_perf, label="train")
    ax.plot(val_perf, label="val")
    ax.legend()

    plt.ylabel(y_label)
    plt.xlabel("Epochs")

    plt.show()


def beam_search(model: torch.nn.Module, init_tokens: list, beam_width: int=3, max_len: int=5) -> torch.Tensor:
    """
    A simple beam search implementation for text generation.
    :param model: A recurrent model that outputs a log probability distribution over the entire vocabulary
    :param init_tokens: The context tokens before the target token(s) we want to generate, i.e. the start of the sentence.
    :beam_width: The size of the beam (k). We select the beam_width number of tokens with the highest predicted (log) probabilities.
    :param max_len: The maximum length of the generated sequence/sentence.
    """
    
    # Prepares initial tokens for input to the model
    torch.LongTensor(init_tokens).unsqueeze(0)  # Convert to tensor, add batch to input dimensions

    # if isinstance(model, GenerativeLSTM):
    # Initial hidden state
    hidden = None

    # Gets the hidden state for the intial tokens
    with torch.no_grad():
        for token in init_tokens[0]:
            _, hidden = model(torch.LongTensor([token]).unsqueeze(0), hidden)
    
    # Gives intital tokens a candidate score of 0
    sequences = [(init_tokens, 0.0)]

    # Main loop
    for i in range(max_len):
        candidates = []  # Keeps track of candidate tokens
        for seq, seq_score in sequences:
            input_token = torch.LongTensor(seq[-1])
            with torch.no_grad():
                out, hidden = model(input_token, hidden)
            softmax_scores = torch.log_softmax(out.squeeze(0), dim=0)
            top_k = torch.topk(softmax_scores, beam_width)

            for i in range(beam_width):
                token, token_score = top_k[i][0].item(), top_k[i][1].item()
                candidate = seq.extend(token)
                cand_score = token_score + seq_score
                candidates.append((candidate, cand_score))
    
    # TESTING ONLY
    return candidates