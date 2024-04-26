import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tempfile
from filelock import FileLock

from .models import CBOW
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


def load_data(data_path):
    with FileLock(os.path.expanduser("~/.data_lock")):
        train_set = torch.load(data_path + "/data_train.pt")
        val_set = torch.load(data_path + "/data_val.pt")
        test_set = torch.load(data_path + "/data_test.pt")
        weights = torch.load(data_path + "/class_weights.pt")

    return train_set, val_set, test_set, weights


def train_embedding(config):
    # hardcoded values for testing
    net = CBOW(46, 6, config["embedding_dim"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    net.to(device)

    train_set, val_set, _, weights = load_data(config["cwd"] + "/generated_data")
    weights = weights.to(device)

    criterion = nn.NLLLoss(weight=weights)
    optimizer = Adam(net.parameters(), lr=config["lr"])

    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_loader = DataLoader(train_set, batch_size=int(config["batch_size"]))
    val_loader = DataLoader(val_set, batch_size=int(config["batch_size"]))

    for epoch in range(config["epochs"]):
        running_loss = 0.0
        epoch_steps = 0

        for i, (contexts, targets) in enumerate(train_loader):
            contexts, targets = contexts.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(contexts)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1

            if i % 2000 == 1999:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/epoch_steps:.3f}")
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0

        for contexts, targets in val_loader:
            with torch.no_grad():
                contexts, targets = contexts.to(device), targets.to(device)
                outputs = net(contexts)

                predictions = torch.argmax(outputs, dim=1)

                total += targets.shape[0]
                correct += int((predictions == targets).sum())

                loss = criterion(outputs, targets)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def test_best_model(best_result, cwd):
    # change to able to use different model architectures
    best_trained_model = CBOW(46, 6, best_result.config["embedding_dim"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(
        best_result.checkpoint.to_directory(), "checkpoint.pt"
    )

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    _, _, test_set, _ = load_data(cwd + "/generated_data")
    test_loader = DataLoader(test_set, batch_size=best_result.config["batch_size"])

    correct = 0
    total = 0

    with torch.no_grad():
        for context, target in test_loader:
            context, target = context.to(device), target.to(device)
            outputs = best_trained_model(context)
            predictions = torch.argmax(outputs, dim=1)

            total += target.shape[0]
            correct += int((predictions == target).sum())

    print(f"Best trial test set accuracy: {correct/total}")


def main(config, num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    os.environ["PYTHONPATH"] = config["cwd"]
    ray.init(log_to_driver=False)
    config["epochs"] = max_num_epochs
    scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_embedding),
            resources={"cpu": 10, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(verbose=1, log_to_file=False),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_result.metrics['accuracy']}")

    test_best_model(best_result, config["cwd"])


if __name__ == "__main__":
    # print(os.getcwd())
    # print(os.path.isfile("./generated_data/class_weights.pt"))
    cwd = os.getcwd()
    config = {
        # "arch": tune.grid_search(["SimpleMLP", "AttentionMLP", "ConjugationRNN"]),
        "cwd": cwd,
        "vocab_size": 46,
        "embedding_dim": 12,
        "arch": tune.grid_search(["CBOW"]),  # can be added as parameter later
        "embedding_dim": tune.choice([4, 6]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([64, 32]),
    }
    main(cwd, 10, 10)
