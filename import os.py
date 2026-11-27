import os
import numpy as np


class FixedBenchmarkMLP:
    """
    Fixed 3-Layer MLP (784 -> 256 -> 128 -> 10) for optimizer benchmarking.
    Exposes self.params and self.grads dictionaries so external optimizers
    can update weights cleanly without modifying model internals.
    """

    def __init__(self, layer_dims=(784, 256, 128, 10), seed=42):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # 3 weight layers
        self.seed = seed

        self.params = {}
        self.grads = {}
        self.cache = {}

        self._initialize_weights(seed)

    def _initialize_weights(self, seed: int):
        """Deterministic He initialization for ReLU layers, Xavier for Output."""
        rng = np.random.default_rng(seed)

        for l in range(1, self.num_layers + 1):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]

            if l < self.num_layers:
                # He Normal initialization for ReLU hidden layers
                std = np.sqrt(2.0 / n_in)
            else:
                # Xavier/Glorot Normal initialization for Softmax output layer
                std = np.sqrt(1.0 / n_in)

            self.params[f"W{l}"] = rng.normal(0.0, std, size=(n_in, n_out)).astype(np.float32)
            self.params[f"b{l}"] = np.zeros((1, n_out), dtype=np.float32)
            self.grads[f"W{l}"] = np.zeros_like(self.params[f"W{l}"])
            self.grads[f"b{l}"] = np.zeros_like(self.params[f"b{l}"])

    @staticmethod
    def _relu(Z):
        return np.maximum(0.0, Z)

    @staticmethod
    def _relu_backward(dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0.0
        return dZ

    @staticmethod
    def _softmax(Z):
        # Shift by row max for numerical stability against overflow
        shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass returning class probability distributions (N, 10)."""
        self.cache["A0"] = X
        A = X

        # Hidden layers (ReLU)
        for l in range(1, self.num_layers):
            Z = A @ self.params[f"W{l}"] + self.params[f"b{l}"]
            A = self._relu(Z)
            self.cache[f"Z{l}"] = Z
            self.cache[f"A{l}"] = A

        # Output layer (Softmax)
        L = self.num_layers
        ZL = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        AL = self._softmax(ZL)
        self.cache[f"Z{L}"] = ZL
        self.cache[f"A{L}"] = AL
        return AL

    def compute_loss(self, probs: np.ndarray, y_one_hot: np.ndarray) -> float:
        """Categorical Cross-Entropy Loss."""
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        N = y_one_hot.shape[0]
        return float(-np.sum(y_one_hot * np.log(probs_clipped)) / N)

    def backward(self, y_one_hot: np.ndarray):
        """Computes gradients via backpropagation and stores them in self.grads."""
        N = y_one_hot.shape[0]
        L = self.num_layers

        # Derivative of Categorical Cross-Entropy + Softmax simplifies to (P - Y) / N
        dZ = (self.cache[f"A{L}"] - y_one_hot) / N

        for l in reversed(range(1, L + 1)):
            A_prev = self.cache[f"A{l - 1}"]
            self.grads[f"W{l}"] = A_prev.T @ dZ
            self.grads[f"b{l}"] = np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = dZ @ self.params[f"W{l}"].T
                dZ = self._relu_backward(dA_prev, self.cache[f"Z{l - 1}"])

    def save_initial_weights(self, filepath: str = "./checkpoints/initial_mlp_weights.npz"):
        """Serializes starting weights to disk so every optimizer starts identically."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, **self.params)
        print(f"Serialized initial MLP weights to {filepath}")

    def load_initial_weights(self, filepath: str = "./checkpoints/initial_mlp_weights.npz"):
        """Restores exact starting weights before running a new optimizer."""
        loaded = np.load(filepath)
        for key in self.params:
            self.params[key] = loaded[key].copy()
            self.grads[key].fill(0.0)
        print(f"Loaded locked initial weights from {filepath}")


if __name__ == "__main__":
    # 1. Instantiate model with fixed seed and save starting point to disk
    model = FixedBenchmarkMLP(layer_dims=(784, 256, 128, 10), seed=42)
    model.save_initial_weights()

    # 2. Verify forward/backward shapes and deterministic reload
    dummy_X = np.random.randn(8, 784).astype(np.float32)
    dummy_y = np.eye(10, dtype=np.float32)[np.arange(8)]

    probs = model.forward(dummy_X)
    loss = model.compute_loss(probs, dummy_y)
    model.backward(dummy_y)

    print(f"Output shape: {probs.shape} | Row sum check: {probs[0].sum():.5f} | Initial Loss: {loss:.4f}")