"""HMM Viterbi post-processing for sleep stage smoothing."""

import numpy as np
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
