"""Post-processing: HMM Viterbi smoothing + logit bias threshold optimization."""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


class HMMPostProcessor:
    """Posterior-based Viterbi smoothing for sleep stage sequences.

    Two improvements over a naive CategoricalHMM:
      1. Operates on posterior probabilities (softmax of logits), not on
         argmax integers. This is what AttnSleep and SleepTransformer do.
         Argmax + identity-like emission collapses the smoother because
         Viterbi just rubber-stamps the predictions.
      2. Transition matrix is fitted with Laplace smoothing from the
         training label sequence. If `recording_lengths` is provided to
         fit(), cross-recording transitions are excluded.
    """

    def __init__(self, num_classes: int = 5, smoothing: float = 1.0):
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.log_trans: np.ndarray | None = None
        self.log_start: np.ndarray | None = None
        self.log_class_prior: np.ndarray | None = None

    def fit(
        self,
        labels: np.ndarray,
        recording_lengths: list[int] | None = None,
    ) -> "HMMPostProcessor":
        """Estimate transition matrix from training labels.

        Args:
            labels: (N,) training labels in temporal order
            recording_lengths: optional list of per-recording lengths so that
                cross-recording transitions are excluded. If None, the whole
                array is treated as one continuous sequence (slight bias).

        Returns:
            self
        """
        K = self.num_classes
        trans = np.full((K, K), self.smoothing, dtype=np.float64)

        if recording_lengths is None:
            for i in range(len(labels) - 1):
                trans[labels[i], labels[i + 1]] += 1
        else:
            offset = 0
            for n in recording_lengths:
                seq = labels[offset:offset + n]
                for i in range(len(seq) - 1):
                    trans[seq[i], seq[i + 1]] += 1
                offset += n

        trans /= trans.sum(axis=1, keepdims=True)
        self.log_trans = np.log(trans)

        # Initial state distribution from class frequencies
        counts = np.bincount(labels, minlength=K).astype(np.float64) + self.smoothing
        prior = counts / counts.sum()
        self.log_start = np.log(prior)
        self.log_class_prior = np.log(prior)

        return self

    def smooth_posteriors(
        self,
        log_posteriors: np.ndarray,
        recording_lengths: list[int] | None = None,
    ) -> np.ndarray:
        """Viterbi decoding over posterior log-probabilities.

        Uses the posteriors directly as emission scores — this is the
        practical convention used by AttnSleep / SleepTransformer. We avoid
        a Bayes prior correction because it destabilises Viterbi on very
        imbalanced datasets (rare classes with tiny priors get amplified
        out of proportion to the modest evidence in the posteriors).

        Args:
            log_posteriors: (N, K) log P(state | observation) per epoch.
                Typically log_softmax(logits).
            recording_lengths: optional per-recording lengths so Viterbi is
                run independently on each recording (no smoothing across
                subject/recording boundaries).

        Returns:
            smoothed_predictions: (N,) integer class predictions
        """
        if self.log_trans is None:
            raise RuntimeError("Call fit() before smooth_posteriors()")

        log_emit = log_posteriors  # (N, K) used directly

        if recording_lengths is None:
            return self._viterbi(log_emit)

        out = np.empty(log_emit.shape[0], dtype=np.int64)
        offset = 0
        for n in recording_lengths:
            out[offset:offset + n] = self._viterbi(log_emit[offset:offset + n])
            offset += n
        return out

    def _viterbi(self, log_emit: np.ndarray) -> np.ndarray:
        """Run Viterbi on a single sequence of (T, K) log emissions."""
        T, K = log_emit.shape
        delta = np.full((T, K), -np.inf, dtype=np.float64)
        psi = np.zeros((T, K), dtype=np.int64)

        delta[0] = self.log_start + log_emit[0]
        for t in range(1, T):
            # scores[i, j] = delta[t-1, i] + log_trans[i, j]
            scores = delta[t - 1, :, None] + self.log_trans  # (K, K)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(K)] + log_emit[t]

        # Backtrack
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    # Backwards-compatible API: decode hard predictions by treating argmax
    # as a one-hot posterior. Prefer smooth_posteriors() for real gains.
    def decode(self, predictions: np.ndarray) -> np.ndarray:
        K = self.num_classes
        N = len(predictions)
        log_post = np.full((N, K), np.log(0.05 / (K - 1)), dtype=np.float64)
        log_post[np.arange(N), predictions] = np.log(0.95)
        return self.smooth_posteriors(log_post)


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
