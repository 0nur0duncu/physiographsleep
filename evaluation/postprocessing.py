"""Post-processing: HMM Viterbi smoothing + logit bias threshold optimization."""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from hmmlearn.hmm import CategoricalHMM


class HMMPostProcessor:
    """Learn transition probabilities from training data and apply Viterbi decoding."""

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.model: CategoricalHMM | None = None

    def fit(self, labels: np.ndarray) -> "HMMPostProcessor":
        """Fit HMM from training label sequences.

        Args:
            labels: (N,) training labels in recording order

        Returns:
            self
        """
        self.model = CategoricalHMM(
            n_components=self.num_classes,
            n_features=self.num_classes,
            n_iter=100,
            random_state=42,
        )

        # Start probability from first label
        start_prob = np.ones(self.num_classes) / self.num_classes
        self.model.startprob_ = start_prob

        # Transition matrix from consecutive pairs
        trans = np.ones((self.num_classes, self.num_classes)) * 1e-6
        for i in range(len(labels) - 1):
            trans[labels[i], labels[i + 1]] += 1
        trans /= trans.sum(axis=1, keepdims=True)
        self.model.transmat_ = trans

        # Emission: identity-like (model output ≈ observation)
        emission = np.eye(self.num_classes) * 0.8 + 0.04
        self.model.emissionprob_ = emission

        return self

    def decode(self, predictions: np.ndarray) -> np.ndarray:
        """Apply Viterbi decoding.

        Args:
            predictions: (N,) predicted class labels

        Returns:
            smoothed: (N,) post-processed labels
        """
        if self.model is None:
            raise RuntimeError("Call fit() before decode()")

        obs = predictions.reshape(-1, 1)
        _, smoothed = self.model.decode(obs, algorithm="viterbi")
        return smoothed


class LogitBiasOptimizer:
    """Find per-class logit biases that maximize macro-F1 on validation data.

    Instead of argmax(logits), we compute argmax(logits + bias).
    A positive bias for class k makes the model more likely to predict k.
    Optimized on validation set via Nelder-Mead to maximize macro-F1.
    """

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.bias: np.ndarray | None = None

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        n_restarts: int = 5,
    ) -> "LogitBiasOptimizer":
        """Optimize biases on validation data.

        Args:
            val_logits: (N, C) raw logits from model on validation set
            val_labels: (N,) ground truth labels
            n_restarts: number of random restarts for optimization

        Returns:
            self
        """
        best_bias = np.zeros(self.num_classes)
        best_mf1 = -1.0

        def neg_macro_f1(bias: np.ndarray) -> float:
            adjusted = val_logits + bias
            preds = adjusted.argmax(axis=1)
            return -f1_score(val_labels, preds, average="macro", zero_division=0)

        # Try multiple restarts
        for i in range(n_restarts):
            if i == 0:
                x0 = np.zeros(self.num_classes)
            else:
                x0 = np.random.uniform(-1.0, 1.0, self.num_classes)

            result = minimize(
                neg_macro_f1, x0,
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5},
            )

            mf1 = -result.fun
            if mf1 > best_mf1:
                best_mf1 = mf1
                best_bias = result.x.copy()

        # Normalize: set first class bias to 0 (relative biases matter)
        best_bias -= best_bias[0]
        self.bias = best_bias

        # Report
        baseline_preds = val_logits.argmax(axis=1)
        baseline_mf1 = f1_score(val_labels, baseline_preds, average="macro", zero_division=0)
        print(f"Logit bias optimization: MF1 {baseline_mf1:.4f} → {best_mf1:.4f} (+{best_mf1-baseline_mf1:.4f})")
        print(f"Optimized biases: {[f'{b:.3f}' for b in self.bias]}")

        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply optimized biases to logits and return predictions.

        Args:
            logits: (N, C) raw logits

        Returns:
            predictions: (N,) adjusted class predictions
        """
        if self.bias is None:
            raise RuntimeError("Call fit() before apply()")
        adjusted = logits + self.bias
        return adjusted.argmax(axis=1)
