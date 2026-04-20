"""Post-processing: HMM Viterbi smoothing + logit bias + temperature scaling."""

import numpy as np
import torch
import torch.nn.functional as F
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


class TemperatureScaling:
    """Post-hoc temperature scaling (Guo et al., ICML 2017).

    Calibrates a trained model by finding a single scalar T > 0 that
    minimises NLL on a held-out validation set.  The calibrated logits
    are ``logits / T``.  Because division by a positive scalar preserves
    argmax, accuracy and macro-F1 are unchanged; only the softmax
    probabilities (and thus NLL / ECE / Brier) improve when the raw
    model is over- or under-confident.

    Usage:
        ts = TemperatureScaling().fit(val_logits, val_labels)
        scaled = ts.apply(test_logits)          # (N, C) calibrated logits
        probs  = ts.predict_proba(test_logits)  # (N, C) calibrated probs
    """

    def __init__(self) -> None:
        self.T: float = 1.0
        self.nll_before: float | None = None
        self.nll_after: float | None = None

    def fit(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        max_iter: int = 200,
    ) -> "TemperatureScaling":
        """Optimise T on validation logits using LBFGS on NLL.

        Args:
            val_logits: (N, C) raw model logits on the validation set
            val_labels: (N,) integer labels
            max_iter:   LBFGS iteration budget

        Returns:
            self
        """
        logits = torch.from_numpy(val_logits).float()
        labels = torch.from_numpy(val_labels).long()

        with torch.no_grad():
            self.nll_before = float(F.cross_entropy(logits, labels).item())

        # log_T is optimised to keep T strictly positive (T = exp(log_T)).
        log_T = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            T = log_T.exp()
            loss = F.cross_entropy(logits / T, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            T = log_T.exp().item()
            self.T = float(T)
            self.nll_after = float(
                F.cross_entropy(logits / log_T.exp(), labels).item()
            )

        print(
            f"Temperature scaling: T={self.T:.4f}  "
            f"NLL {self.nll_before:.4f} → {self.nll_after:.4f} "
            f"({'over-confident' if self.T > 1.0 else 'under-confident' if self.T < 1.0 else 'calibrated'})"
        )
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated logits (logits / T). Argmax is preserved."""
        return logits / self.T

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities (N, C)."""
        return F.softmax(torch.from_numpy(logits / self.T).float(), dim=-1).numpy()


# ------------------------------------------------------------------
# Calibration diagnostics
# ------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
) -> float:
    """Expected Calibration Error (Guo et al., ICML 2017).

    Splits the predicted-confidence axis into ``n_bins`` equal-width bins
    and returns the weighted average of |accuracy − confidence| per bin.
    Lower is better; 0 means perfect calibration.

    Args:
        probs:  (N, C) softmax probabilities
        labels: (N,) integer ground-truth labels
        n_bins: number of confidence bins (default 15, per the paper)

    Returns:
        ece: scalar in [0, 1]
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, C), got shape {probs.shape}")
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Last bin is closed on both sides so confidence == 1.0 is counted
        if hi == 1.0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        n_in = int(in_bin.sum())
        if n_in == 0:
            continue
        bin_acc = float(accuracies[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (n_in / N) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error on one-hot targets).

    Brier = (1/N) · Σ_i Σ_k (p_ik − y_ik)^2

    Lower is better; bounded in [0, 2] for categorical outcomes.

    Args:
        probs:  (N, C) softmax probabilities
        labels: (N,) integer ground-truth labels

    Returns:
        brier: scalar
    """
    N, C = probs.shape
    onehot = np.zeros((N, C), dtype=np.float64)
    onehot[np.arange(N), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())
