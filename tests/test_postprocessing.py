"""Unit tests for HMM Viterbi smoothing and logit bias optimization."""

import numpy as np
import pytest

from physiographsleep.evaluation.postprocessing import (
    HMMPostProcessor,
    LogitBiasOptimizer,
    TemperatureScaling,
    compute_ece,
    compute_brier,
)


def _one_hot(labels: np.ndarray, k: int = 5) -> np.ndarray:
    """Hard one-hot probabilities for a label sequence."""
    p = np.full((len(labels), k), 1e-3)
    p[np.arange(len(labels)), labels] = 1.0 - (k - 1) * 1e-3
    return np.log(p)


def test_hmm_fit_produces_log_trans():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=1000)
    hmm = HMMPostProcessor(num_classes=5).fit(labels)
    assert hmm.log_trans is not None and hmm.log_trans.shape == (5, 5)
    # Rows of trans (exp of log_trans) should sum to 1
    rows = np.exp(hmm.log_trans).sum(axis=1)
    np.testing.assert_allclose(rows, 1.0, atol=1e-8)


def test_hmm_smooths_isolated_label_flips():
    """A single low-confidence wrong prediction should be corrected by Viterbi
    when the transition matrix strongly favors self-transitions."""
    train = np.array([2] * 500 + [3] * 200 + [2] * 300, dtype=np.int64)
    hmm = HMMPostProcessor(num_classes=5).fit(train)

    # Soft posteriors: model is highly confident on class 2 around the flip,
    # but uncertain on the flip itself (only slight preference for class 4).
    seq_len = 41
    log_post = np.full((seq_len, 5), np.log(0.05))
    log_post[:, 2] = np.log(0.80)  # baseline strong belief in class 2
    # Middle epoch: slight preference for class 4 over class 2
    log_post[20, :] = np.log(0.15)
    log_post[20, 4] = np.log(0.40)
    log_post[20, 2] = np.log(0.30)

    smoothed = hmm.smooth_posteriors(log_post)
    assert smoothed[0] == 2 and smoothed[-1] == 2
    assert smoothed[20] == 2, f"Expected isolated soft flip to be smoothed, got {smoothed[20]}"


def test_hmm_does_nothing_with_uniform_transitions():
    """When transitions are uniform, Viterbi must respect emission posteriors."""
    hmm = HMMPostProcessor(num_classes=5)
    # Force uniform transitions
    hmm.log_trans = np.log(np.full((5, 5), 0.2))
    hmm.log_start = np.log(np.full(5, 0.2))
    hmm.log_class_prior = np.log(np.full(5, 0.2))

    rng = np.random.default_rng(1)
    pred_seq = rng.integers(0, 5, size=200).astype(np.int64)
    log_post = _one_hot(pred_seq)
    smoothed = hmm.smooth_posteriors(log_post)
    # Should equal argmax of posteriors when transitions don't matter
    np.testing.assert_array_equal(smoothed, pred_seq)


def test_hmm_per_recording_no_cross_smoothing():
    """smooth_posteriors with recording_lengths must process each segment
    independently."""
    train = np.array([0] * 500 + [1] * 500, dtype=np.int64)
    hmm = HMMPostProcessor(num_classes=5).fit(train)

    # Two recordings: first all zeros, second all ones, glued together
    pred = np.concatenate([np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)])
    log_post = _one_hot(pred)
    out = hmm.smooth_posteriors(log_post, recording_lengths=[50, 50])
    assert (out[:50] == 0).all()
    assert (out[50:] == 1).all()


def test_hmm_backward_compat_decode():
    """Legacy decode(int_predictions) must still work via one-hot conversion."""
    train = np.array([2] * 500 + [3] * 500, dtype=np.int64)
    hmm = HMMPostProcessor(num_classes=5).fit(train)
    pred = np.array([2] * 30 + [3] * 30, dtype=np.int64)
    out = hmm.decode(pred)
    assert out.shape == pred.shape
    assert out.dtype.kind in ("i", "u")


def test_logit_bias_improves_macro_f1():
    """On an imbalanced fake dataset, bias optimization should not regress."""
    rng = np.random.default_rng(7)
    n = 300
    labels = np.concatenate([
        np.zeros(200, dtype=np.int64),
        np.ones(50, dtype=np.int64),
        np.full(50, 2, dtype=np.int64),
    ])
    # Logits favor class 0; minority classes (1, 2) get small noisy bumps
    logits = rng.normal(0, 0.5, size=(n, 5))
    logits[:, 0] += 1.5
    logits[200:250, 1] += 0.8
    logits[250:300, 2] += 0.8

    from sklearn.metrics import f1_score
    base_mf1 = f1_score(labels, logits.argmax(1), average="macro", zero_division=0)

    opt = LogitBiasOptimizer(num_classes=5)
    opt.fit(logits, labels, n_restarts=3)
    new_preds = opt.apply(logits)
    new_mf1 = f1_score(labels, new_preds, average="macro", zero_division=0)

    assert new_mf1 >= base_mf1 - 1e-6, (
        f"Logit bias regressed: {base_mf1:.4f} -> {new_mf1:.4f}"
    )


def _overconfident_logits(rng, n=400, c=5, scale=4.0, err_rate=0.2):
    """Build overconfident logits: labels are ~80% correct but the argmax
    is pushed to ~0.99 confidence via large logit scale."""
    labels = rng.integers(0, c, size=n)
    logits = rng.normal(0, 0.1, size=(n, c))
    # Pick argmax per row: correct with prob (1 - err_rate)
    picks = labels.copy()
    flips = rng.random(n) < err_rate
    # Flip to a different random class
    picks[flips] = (labels[flips] + rng.integers(1, c, size=flips.sum())) % c
    logits[np.arange(n), picks] += scale
    return logits.astype(np.float64), labels.astype(np.int64)


def test_temperature_preserves_argmax():
    rng = np.random.default_rng(3)
    logits, labels = _overconfident_logits(rng)
    ts = TemperatureScaling().fit(logits, labels)
    scaled = ts.apply(logits)
    # Division by positive T cannot change argmax
    assert (scaled.argmax(1) == logits.argmax(1)).all()
    assert ts.T > 0.0


def test_temperature_reduces_nll_on_overconfident_logits():
    rng = np.random.default_rng(4)
    logits, labels = _overconfident_logits(rng, err_rate=0.25)
    ts = TemperatureScaling().fit(logits, labels)
    # Overconfident model → T should push >1 and NLL should drop
    assert ts.T > 1.0, f"Expected T > 1 for overconfident logits, got T={ts.T}"
    assert ts.nll_after <= ts.nll_before + 1e-6


def test_temperature_identity_on_calibrated_logits():
    """If logits are already near-calibrated (small magnitude), T ≈ 1."""
    rng = np.random.default_rng(5)
    n, c = 500, 5
    labels = rng.integers(0, c, size=n)
    logits = rng.normal(0, 0.5, size=(n, c))
    # Give a small correct-class bias so labels are learnable at all
    logits[np.arange(n), labels] += 0.3
    ts = TemperatureScaling().fit(logits.astype(np.float64), labels.astype(np.int64))
    assert 0.5 < ts.T < 2.0, f"Expected T near 1.0, got {ts.T}"


def test_ece_bounds_and_perfect_case():
    rng = np.random.default_rng(6)
    n, c = 200, 5
    labels = rng.integers(0, c, size=n)
    # Perfect one-hot correct predictions → ECE = 0
    perfect = np.full((n, c), 1e-9)
    perfect[np.arange(n), labels] = 1.0 - (c - 1) * 1e-9
    ece = compute_ece(perfect, labels, n_bins=15)
    assert ece < 1e-6, f"Perfect predictions should have ECE≈0, got {ece}"
    # Always-wrong confident predictions → ECE large (close to confidence - 0)
    wrong = np.full((n, c), 1e-9)
    wrong_idx = (labels + 1) % c
    wrong[np.arange(n), wrong_idx] = 1.0 - (c - 1) * 1e-9
    ece_bad = compute_ece(wrong, labels, n_bins=15)
    assert ece_bad > 0.9


def test_brier_bounds_and_perfect_case():
    rng = np.random.default_rng(7)
    n, c = 200, 5
    labels = rng.integers(0, c, size=n)
    perfect = np.full((n, c), 1e-9)
    perfect[np.arange(n), labels] = 1.0 - (c - 1) * 1e-9
    brier = compute_brier(perfect, labels)
    assert brier < 1e-6
    # Uniform predictions on 5 classes: Brier = (4/5)^2 + 4*(1/5)^2 = 0.64 + 0.16 = 0.8
    uniform = np.full((n, c), 1.0 / c)
    brier_u = compute_brier(uniform, labels)
    np.testing.assert_allclose(brier_u, 0.8, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
