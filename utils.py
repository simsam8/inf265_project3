import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt


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
