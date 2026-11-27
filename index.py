import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def prepare_and_save_dataset(
    dataset_name: str = "Fashion-MNIST",
    output_dir: str = "./processed_data",
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Downloads, validates, preprocesses, splits, and serializes an image dataset
    so all teammates use identical, deterministic partitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name.lower()}_splits.npz")

    # 1. DOWNLOAD & LOAD RAW DATA
    # Fashion-MNIST (70,000 grayscale 28x28 images across 10 classes)
    print(f"Downloading {dataset_name}...")
    X, y = fetch_openml(
        dataset_name, version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    y = y.astype(np.int64)

    # 2. SANITY CHECKS (Before Preprocessing)
    assert not np.isnan(X).any(), "Error: Dataset contains NaN pixel values!"
    assert not np.isnan(y).any(), "Error: Dataset contains NaN labels!"
    assert X.shape[0] == y.shape[0], "Error: Sample count mismatch between X and y!"

    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Raw Shape: {X.shape} | Dtype: {X.dtype}")
    print(f"Pixel Range: [{X.min()}, {X.max()}]")
    print(f"Class Distribution: {dict(zip(unique_classes, class_counts))}")

    # 3. PREPROCESSING
    # Standardize pixel intensities: [0, 255] -> [0, 1] -> zero-mean, unit-variance
    X = X.astype(np.float32) / 255.0

    # Keep 70,000 standard split: first 60k for train/val, last 10k locked as Test Set
    X_train_full, X_test = X[:60000], X[60000:]
    y_train_full, y_test = y[:60000], y[60000:]

    # Compute mean and std ONLY on training data to prevent data leakage
    mean = X_train_full.mean(axis=0, keepdims=True)
    std = X_train_full.std(axis=0, keepdims=True) + 1e-8  # epsilon prevents div-by-zero

    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std

    # 4. DETERMINISTIC STRATIFIED SPLIT (Train vs. Validation)
    # Stratify guarantees each split preserves the exact 10% class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_train_full,
    )

    # One-hot encode labels for Categorical Cross-Entropy: shape (N, 10)
    num_classes = len(unique_classes)
    y_train_oh = np.eye(num_classes, dtype=np.float32)[y_train]
    y_val_oh = np.eye(num_classes, dtype=np.float32)[y_val]
    y_test_oh = np.eye(num_classes, dtype=np.float32)[y_test]

    # 5. SAVE LOCKED PARTITIONS TO DISK
    np.savez_compressed(
        save_path,
        X_train=X_train,
        y_train=y_train_oh,
        y_train_idx=y_train,
        X_val=X_val,
        y_val=y_val_oh,
        y_val_idx=y_val,
        X_test=X_test,
        y_test=y_test_oh,
        y_test_idx=y_test,
        seed=seed,
    )
    print(f"\nSaved locked splits to {save_path}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


def load_locked_dataset(filepath: str):
    """Helper function imported by the training pipeline."""
    data = np.load(filepath)
    return {
        "train": (data["X_train"], data["y_train"], data["y_train_idx"]),
        "val": (data["X_val"], data["y_val"], data["y_val_idx"]),
        "test": (data["X_test"], data["y_test"], data["y_test_idx"]),
    }


if __name__ == "__main__":
    prepare_and_save_dataset()