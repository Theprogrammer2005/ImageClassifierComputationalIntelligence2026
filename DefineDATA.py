from abc import ABC, abstractmethod
import csv
import json
import os
import time
from typing import Any, Callable, Dict, List
import numpy as np


# ==============================================================================
# 1. STRATEGY INTERFACE (Abstract Base Class)
# ==============================================================================
class BaseOptimizer(ABC):
    """
    Uniform contract that every teammate's optimizer MUST implement.
    The training loop interacts ONLY with this interface.
    """

    def __init__(self, name: str, lr: float):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.name = name
        self.lr = lr

    @abstractmethod
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Applies an in-place parameter update using the current gradients.
        Must NOT modify grads or return a new parameter dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clears all internal state buffers (momentum, moments, step counters)."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Returns hyperparameters for logging and reproducibility."""
        return {"optimizer_name": self.name, "lr": self.lr}


# ==============================================================================
# 2. CONCRETE STRATEGIES (Reference Implementations for Teammates)
# ==============================================================================
class SGDStrategy(BaseOptimizer):
    """Vanilla Mini-Batch Stochastic Gradient Descent."""

    def __init__(self, lr: float = 0.01):
        super().__init__(name="SGD", lr=lr)

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        for key in params:
            params[key] -= self.lr * grads[key]

    def reset(self) -> None:
        pass  # Stateless optimizer


class MomentumStrategy(BaseOptimizer):
    """SGD with Classical Momentum."""

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(name="Momentum", lr=lr)
        self.beta = beta
        self.velocity: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.beta * self.velocity[key] + grads[key]
            params[key] -= self.lr * self.velocity[key]

    def reset(self) -> None:
        self.velocity.clear()

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "beta": self.beta}


class RMSpropStrategy(BaseOptimizer):
    """Root Mean Square Propagation."""

    def __init__(self, lr: float = 0.001, decay_rate: float = 0.9, eps: float = 1e-8):
        super().__init__(name="RMSprop", lr=lr)
        self.decay_rate = decay_rate
        self.eps = eps
        self.sq_grads: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        for key in params:
            if key not in self.sq_grads:
                self.sq_grads[key] = np.zeros_like(params[key])
            self.sq_grads[key] = (
                self.decay_rate * self.sq_grads[key]
                + (1.0 - self.decay_rate) * (grads[key] ** 2)
            )
            params[key] -= self.lr * grads[key] / (np.sqrt(self.sq_grads[key]) + self.eps)

    def reset(self) -> None:
        self.sq_grads.clear()

    def get_config(self) -> Dict[str, Any]:
        return {**super().get_config(), "decay_rate": self.decay_rate, "eps": self.eps}


class AdamStrategy(BaseOptimizer):
    """Adaptive Moment Estimation (Adam) with Bias Correction."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(name="Adam", lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t: int = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self.m.clear()
        self.v.clear()
        self.t = 0

    def get_config(self) -> Dict[str, Any]:
        return {
            **super().get_config(),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }


# ==============================================================================
# 3. SCHEMA-ENFORCED STRUCTURED TELEMETRY LOGGER
# ==============================================================================
class ExperimentLogger:
    """
    Guarantees every optimizer run outputs identical CSV and JSON schemas
    so Phase 5 plotting and comparison scripts require zero manual cleaning.
    """

    SCHEMA_FIELDS = [
        "run_id",
        "optimizer",
        "lr",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "epoch_time_sec",
        "cumulative_time_sec",
    ]

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self._cumulative_time = 0.0

    def start_run(self) -> None:
        self.records.clear()
        self._cumulative_time = 0.0

    def log_epoch(
        self,
        run_id: str,
        optimizer: BaseOptimizer,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time_sec: float,
    ) -> Dict[str, Any]:
        self._cumulative_time += epoch_time_sec
        row = {
            "run_id": run_id,
            "optimizer": optimizer.name,
            "lr": optimizer.lr,
            "epoch": int(epoch),
            "train_loss": round(float(train_loss), 6),
            "train_acc": round(float(train_acc), 6),
            "val_loss": round(float(val_loss), 6),
            "val_acc": round(float(val_acc), 6),
            "epoch_time_sec": round(float(epoch_time_sec), 4),
            "cumulative_time_sec": round(float(self._cumulative_time), 4),
        }
        self.records.append(row)
        return row

    def export(self, run_id: str, optimizer: BaseOptimizer) -> None:
        csv_path = os.path.join(self.log_dir, f"{run_id}.csv")
        json_path = os.path.join(self.log_dir, f"{run_id}.json")

        # Export flat CSV for tabular aggregation
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.SCHEMA_FIELDS)
            writer.writeheader()
            writer.writerows(self.records)

        # Export rich JSON with embedded optimizer hyperparameters
        payload = {
            "run_id": run_id,
            "hyperparameters": optimizer.get_config(),
            "total_time_sec": round(self._cumulative_time, 4),
            "history": self.records,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ==============================================================================
# 4. DECOUPLED CONTEXT / BENCHMARK RUNNER
# ==============================================================================
class BenchmarkHarness:
    """
    Executes identical training loops across any number of injected optimizers
    without modifying the model architecture or training engine.
    """

    def __init__(
        self,
        model,
        train_data: tuple,
        val_data: tuple,
        initial_weights_path: str,
        eval_fn: Callable,
        batch_size: int = 64,
        num_epochs: int = 20,
        seed: int = 42,
        log_dir: str = "./logs",
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.initial_weights_path = initial_weights_path
        self.eval_fn = eval_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.logger = ExperimentLogger(log_dir=log_dir)

    @staticmethod
    def _validate_strategy(optimizer: BaseOptimizer) -> None:
        if not isinstance(optimizer, BaseOptimizer):
            raise TypeError(
                f"Optimizer {type(optimizer).__name__} must inherit from BaseOptimizer."
            )

    def run(self, run_id: str, optimizer: BaseOptimizer) -> List[Dict[str, Any]]:
        self._validate_strategy(optimizer)

        # Reset model weights AND optimizer state to guarantee identical starting conditions
        self.model.load_initial_weights(self.initial_weights_path)
        optimizer.reset()
        self.logger.start_run()

        X_train, y_train_oh, y_train_idx = self.train_data
        X_val, y_val_oh, _ = self.val_data
        num_train = X_train.shape[0]
        rng = np.random.default_rng(self.seed)

        print(f"\n=== Benchmarking [{run_id}] ({optimizer.get_config()}) ===")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.perf_counter()

            perm = rng.permutation(num_train)
            X_shuf, y_oh_shuf, y_idx_shuf = (
                X_train[perm],
                y_train_oh[perm],
                y_train_idx[perm],
            )

            running_loss = 0.0
            running_correct = 0

            for start in range(0, num_train, self.batch_size):
                end = min(start + self.batch_size, num_train)
                bs = end - start

                probs = self.model.forward(X_shuf[start:end])
                loss = self.model.compute_loss(probs, y_oh_shuf[start:end])
                self.model.backward(y_oh_shuf[start:end])

                # Polymorphic Strategy call — completely decoupled from update math
                optimizer.step(self.model.params, self.model.grads)

                running_loss += loss * bs
                running_correct += int(np.sum(np.argmax(probs, axis=1) == y_idx_shuf[start:end]))

            train_loss = running_loss / num_train
            train_acc = running_correct / num_train
            val_loss, val_acc = self.eval_fn(self.model, X_val, y_val_oh)
            elapsed = time.perf_counter() - t0

            record = self.logger.log_epoch(
                run_id=run_id,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time_sec=elapsed,
            )

            print(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {record['train_loss']:.4f} | "
                f"Val Loss: {record['val_loss']:.4f} | "
                f"Val Acc: {record['val_acc']*100:.2f}% | "
                f"Time: {record['epoch_time_sec']:.2f}s"
            )

        self.logger.export(run_id=run_id, optimizer=optimizer)
        return list(self.logger.records)