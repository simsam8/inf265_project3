import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    """
    Computes the accuracy of a model on a given dataset.
    """
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


def train(
    epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device=None,
) -> tuple[list, list, list, list]:
    """
    Trains a model with given optimizer and loss function.
    Tracks training and validation loss and accuracy.
    """
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

        if epoch == 1 or epoch % 10 == 0:
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
    f_name=None,
    save=False,
) -> None:
    """
    Creates a plot of training and validation loss/performance over time.
    """
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(train_perf, label="train")
    ax.plot(val_perf, label="val")
    ax.legend()
    plt.ylabel(y_label)
    plt.xlabel("Epochs")
    if save:
        plt.savefig(f_name)
    plt.show()


def plot_training_times(architecture_times, labels, f_name=None, save=False) -> None:
    """
    Creates a plot of average architecture training times.
    """
    _, ax = plt.subplots()
    ax.set_title("Average training time of different architectures")
    ax.bar(labels, architecture_times)
    plt.ylabel("Time (seconds)")
    plt.xlabel("Architecture")
    if save:
        plt.savefig(f_name)
    plt.show()


def beam_search(
    model: nn.Module,
    init_token_indeces: list,
    beam_width: int = 3,
    max_len: int = 5,
    gen_sequences: int = 1,
    print_search_tree: bool = False,
    length_penalty: float = 0.4
) -> list:
    """
    A simple beam search implementation for text generation.
    :param model: A recurrent model that outputs a log probability distribution over the entire vocabulary
    :param init_token_indeces: A list of the context tokens' indeces before the target token(s) we want to generate, i.e. the start of the sentence / prompt.
    :beam_width: The size of the beam (k). We select the beam_width number of tokens with the highest predicted (log) probabilities.
    :param max_len: The maximum length of the generated sequence/sentence.
    :param gen_sequences: The number of generated sequences to return. Returns top gen_sequences candidates from search tree.
    :param print_search_tree: Whether to print all candidates during the search (for debugging/reporting purposes)
    :param length_penalty: The penalty used for length normalization. Lower penalty results in shorter sequences
    :return: A list of (sequence, sequence score) tuples.
    """
    # Stores model to device for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Freeze model weights for inference

    # Gives intital sequence (prompt) a candidate score of 0
    sequences = [(init_token_indeces, 0.0)]
    all_sequences = []
    mapping = torch.load("../generated_data/mapping.pt")

    # Search loop
    for i in range(max_len):
        if print_search_tree:
            print(f"----- Sequences of length {len(init_token_indeces) + i} ------")
        candidates = []  # Keeps track of candidate tokens
        for seq, seq_score in sequences:
            # Converts to Tensor with batch dimension
            input_seq = torch.LongTensor(seq).unsqueeze(0).to(device)
            # Runs inference
            with torch.no_grad():  # Avoid calculating gradients
                out = model(input_seq)
                log_probs = F.log_softmax(out, dim=-1)

            # Gets the index of the top candidates (highest log probability)
            top_k = torch.topk(log_probs, beam_width)

            # Keeps track of candidates
            for j in range(beam_width):
                token_index, token_score = (
                    top_k.indices[0][j].item(),
                    top_k.values[0][j].item(),
                )
                candidate_seq = seq + [token_index]
                # Perform length normalization to avoid punishment of long sequences
                normalized_score = (token_score + seq_score) / len(candidate_seq)**length_penalty
                candidates.append((candidate_seq, normalized_score))
                if print_search_tree:
                    print(
                        f"{[mapping[token_index] for token_index in candidate_seq]} Score: {normalized_score}"
                    )

        # Pruning
        ordered = sorted(candidates, key=lambda c: c[1], reverse=True)
        all_sequences += ordered  # Keep track of all candidates we have used in the search
        sequences = ordered[:beam_width]

    # The sequence(s) with the highest score(s) across all evaluated sequences is returned
    all_sequences_ordered = sorted(all_sequences, key=lambda c: c[1], reverse=True)
    
    return all_sequences_ordered[gen_sequences - 1]
