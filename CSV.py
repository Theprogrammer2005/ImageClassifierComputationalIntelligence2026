import csv
import json
import os
import time
import numpy as np


class BaseNumPyOptimizer:
    """
    Phase 4 Interface Contract: Every teammate's NumPy optimizer must inherit
    from this class and implement step(params, grads).
    """

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params: dict, grads: dict):
        raise NotImplementedError("Subclasses must implement step(params, grads).")

    def reset(self):
        """Optional hook to clear momentum/velocity buffers between runs."""
        pass


class SGDNumPy(BaseNumPyOptimizer):
    """Reference baseline optimizer demonstrating the interface."""

    def step(self, params: dict, grads: dict):
        for key in params:
            params[key] -= self.lr * grads[key]


def evaluate_numpy_model(model, X: np.ndarray, y_one_hot: np.ndarray, batch_size: int = 256):
    """Evaluates loss and classification accuracy in inference mode (no backprop)."""
    num_samples = X.shape[0]
    total_loss = 0.0
    total_correct = 0
    y_true_idx = np.argmax(y_one_hot, axis=1)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        X_batch = X[start:end]
        y_batch_oh = y_one_hot[start:end]
        y_batch_idx = y_true_idx[start:end]

        probs = model.forward(X_batch)
        batch_loss = model.compute_loss(probs, y_batch_oh)

        total_loss += batch_loss * (end - start)
        preds = np.argmax(probs, axis=1)
        total_correct += int(np.sum(preds == y_batch_idx))

    return total_loss / num_samples, total_correct / num_samples


def run_numpy_training_engine(
    model,
    optimizer: BaseNumPyOptimizer,
    train_data: tuple,
    val_data: tuple,
    initial_weights_path: str,
    run_name: str,
    batch_size: int = 64,
    num_epochs: int = 20,
    seed: int = 42,
    log_dir: str = "./logs",
):
    """
    Executes a deterministic training run for a single optimizer and saves
    epoch-level telemetry to CSV and JSON files.
    """
    os.makedirs(log_dir, exist_ok=True)

    # 1. Reset model to the exact locked starting point
    model.load_initial_weights(initial_weights_path)
    optimizer.reset()

    X_train, y_train_oh, y_train_idx = train_data
    X_val, y_val_oh, _ = val_data
    num_train = X_train.shape[0]

    # Seeded RNG guarantees every optimizer receives identical mini-batch sequences
    rng = np.random.default_rng(seed)
    history = []

    print(f"\n--- Starting Run: {run_name} | Batch Size: {batch_size} | Epochs: {num_epochs} ---")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()

        # 2. Shuffle training data deterministically at the start of each epoch
        perm = rng.permutation(num_train)
        X_shuffled = X_train[perm]
        y_oh_shuffled = y_train_oh[perm]
        y_idx_shuffled = y_train_idx[perm]

        running_train_loss = 0.0
        running_train_correct = 0

        # 3. Iterate through mini-batches
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            current_bs = end - start

            X_batch = X_shuffled[start:end]
            y_batch_oh = y_oh_shuffled[start:end]
            y_batch_idx = y_idx_shuffled[start:end]

            # Forward pass
            probs = model.forward(X_batch)
            batch_loss = model.compute_loss(probs, y_batch_oh)

            # Backward pass (populates model.grads)
            model.backward(y_batch_oh)

            # Delegated parameter update (loop does not touch weights directly)
            optimizer.step(model.params, model.grads)

            # Sample-weighted metric accumulation
            running_train_loss += batch_loss * current_bs
            preds = np.argmax(probs, axis=1)
            running_train_correct += int(np.sum(preds == y_batch_idx))

        train_loss = running_train_loss / num_train
        train_acc = running_train_correct / num_train

        # 4. End-of-epoch validation in inference mode
        val_loss, val_acc = evaluate_numpy_model(model, X_val, y_val_oh, batch_size=256)
        epoch_duration = time.perf_counter() - epoch_start

        # 5. Record structured epoch metrics
        record = {
            "optimizer": run_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_time_sec": round(epoch_duration, 4),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_duration:.2f}s] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

    # 6. Export structured logs (JSON + CSV)
    json_path = os.path.join(log_dir, f"{run_name}_metrics.json")
    csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved logs to {json_path} and {csv_path}")
    return history