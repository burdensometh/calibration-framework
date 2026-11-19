# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Tests for the label observation mask contract used by independent (multi-label)
confidence calibration.

Covered properties:

- unknown labels are never treated as negative labels,
- every class is fitted only from its own observed labels and rows are not dropped
  globally (known labels of other classes survive),
- per-class support and unfittable classes are recorded,
- ``transform`` keeps the full class axis with an explicit fallback for
  unfittable classes,
- the observation mask is metadata only and never becomes part of the features,
- the calibration metrics use the same observation mask to decide every
  per-class denominator.
"""

import numpy as np
import pytest

from netcal import (
    LabelObservationError,
    prepare_label_matrix,
    class_status,
)
from netcal.binning import BBQ, ENIR, HistogramBinning, IsotonicRegression
from netcal.metrics import ACE, ECE, MCE
from netcal.scaling import BetaCalibration, LogisticCalibration, TemperatureScaling


def _multilabel_data(n=240, k=3, seed=0):
    """Create independent multi-label data where unknown labels are encoded as NaN."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, k)

    y = np.full((n, k), np.nan)
    y[:, 0] = (rng.rand(n) < X[:, 0]).astype(float)
    y[:80, 1] = (rng.rand(80) < X[:80, 1]).astype(float)
    # class 2 is entirely unobserved

    return X, y


# ---------------------------------------------------------------------------
# label / mask validation
# ---------------------------------------------------------------------------

def test_unknown_labels_are_not_negatives():
    y = np.array([[1.0, np.nan, 0.0], [0.0, 1.0, np.nan]])

    observation = prepare_label_matrix(y)

    # the NaNs survive as NaN and their entries are masked out - never filled with zero
    assert np.isnan(observation.labels[0, 1])
    assert not observation.mask[0, 1]
    np.testing.assert_array_equal(observation.support, [2, 1, 1])


def test_explicit_mask_and_nan_agree():
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mask = np.array([[True, True], [True, False], [False, True]])

    from_mask = prepare_label_matrix(np.zeros_like(y), mask)
    from_nan = prepare_label_matrix(
        np.array([[1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]])
    )

    np.testing.assert_array_equal(from_mask.mask, from_nan.mask)
    np.testing.assert_array_equal(from_mask.support, from_nan.support)


def test_mask_marks_unobserved_entry_as_observed_raises():
    y = np.array([[1.0, np.nan]])
    mask = np.array([[True, True]])

    with pytest.raises(LabelObservationError):
        prepare_label_matrix(y, mask)


def test_mask_shape_mismatch_raises():
    y = np.array([[1.0, np.nan, 0.0]])
    mask = np.ones((1, 4), dtype=bool)

    with pytest.raises(LabelObservationError):
        prepare_label_matrix(y, mask)


def test_observed_non_binary_label_raises():
    y = np.array([[2.0, 0.0]])

    with pytest.raises(LabelObservationError):
        prepare_label_matrix(y)


def test_all_unobserved_raises():
    y = np.full((5, 3), np.nan)

    with pytest.raises(LabelObservationError):
        prepare_label_matrix(y)


def test_class_status_distinguishes_unobserved_and_degenerate():
    y = np.array([
        [1.0, 0.0, np.nan],
        [1.0, 1.0, np.nan],
        [0.0, 1.0, np.nan],
    ])
    observation = prepare_label_matrix(y)

    statuses, support = class_status(observation.labels, observation.mask)

    assert statuses[0] == 'fitted'
    assert statuses[1] == 'fitted'
    assert statuses[2] == 'unobserved'
    np.testing.assert_array_equal(support, [3, 3, 0])

    y_degenerate = np.array([[1.0], [1.0]])
    degenerate = prepare_label_matrix(y_degenerate)
    statuses, _ = class_status(degenerate.labels, degenerate.mask)
    assert statuses[0] == 'degenerate'


# ---------------------------------------------------------------------------
# fitting: per-class use of observed labels only, no global row dropping
# ---------------------------------------------------------------------------

CALIBRATORS = [
    lambda: HistogramBinning(independent_probabilities=True),
    lambda: IsotonicRegression(independent_probabilities=True),
    lambda: BBQ(independent_probabilities=True),
    lambda: ENIR(independent_probabilities=True),
    lambda: LogisticCalibration(independent_probabilities=True),
    lambda: TemperatureScaling(independent_probabilities=True),
    lambda: BetaCalibration(independent_probabilities=True),
]


def _fit(model, X, y, **kwargs):
    if 'scaling' in model.__class__.__module__:
        kwargs.setdefault('tensorboard', False)
    return model.fit(X, y, **kwargs)


@pytest.mark.parametrize("factory", CALIBRATORS)
def test_support_and_unfittable_classes_recorded(factory):
    X, y = _multilabel_data()

    model = _fit(factory(), X, y)

    np.testing.assert_array_equal(model.support_, [240, 80, 0])
    assert model.class_status_ == ['fitted', 'fitted', 'unobserved']
    assert model.fitted_classes_ == [0, 1]
    assert model.fallback_classes_ == [2]
    assert 2 in model.unobserved_classes_


@pytest.mark.parametrize("factory", CALIBRATORS)
def test_full_class_axis_and_explicit_fallback(factory):
    X, y = _multilabel_data()

    model = _fit(factory(), X, y)
    calibrated = model.transform(X)

    # the full class axis is returned ...
    assert calibrated.shape == X.shape

    # ... and the unobserved class explicitly falls back to the uncalibrated confidence
    np.testing.assert_allclose(calibrated[:, 2], X[:, 2], atol=1e-6)


def test_rows_are_not_dropped_globally():
    n, k = 60, 2
    rng = np.random.RandomState(3)
    X = rng.rand(n, k)

    y = np.full((n, k), np.nan)
    y[:30, 0] = (rng.rand(30) < X[:30, 0]).astype(float)
    y[30:, 1] = (rng.rand(30) < X[30:, 1]).astype(float)

    model = HistogramBinning(independent_probabilities=True)
    model.fit(X, y)

    # both classes are fitted from disjoint rows: no sample is dropped in a way that would
    # remove the other class's known labels
    np.testing.assert_array_equal(model.support_, [30, 30])
    assert model.fitted_classes_ == [0, 1]


def test_mask_never_mixed_into_features():
    X, y = _multilabel_data()
    mask = np.isfinite(y)

    with_mask = HistogramBinning(independent_probabilities=True)
    with_mask.fit(X, y, mask=mask)
    from_nan = HistogramBinning(independent_probabilities=True)
    from_nan.fit(X, y)

    # passing the mask explicitly is equivalent to encoding it in the labels; in both cases X
    # keeps exactly the original feature dimension
    assert with_mask.num_classes == X.shape[1]
    np.testing.assert_allclose(with_mask.transform(X), from_nan.transform(X))


def test_masked_array_labels_supported():
    X, y = _multilabel_data()

    masked = np.ma.masked_array(np.nan_to_num(y), mask=~np.isfinite(y))
    model = HistogramBinning(independent_probabilities=True)
    model.fit(X, masked)

    np.testing.assert_array_equal(model.support_, [240, 80, 0])


def test_binary_and_detection_drop_only_unobserved_samples():
    n = 40
    rng = np.random.RandomState(4)

    X = rng.rand(n)
    y = (rng.rand(n) < X).astype(float)
    y[10:20] = np.nan

    model = LogisticCalibration()
    model.fit(X, y, tensorboard=False)

    # the shared binary target is reduced to observed samples only
    assert model.support_.tolist() == [30, 30]
    assert model.transform(X).shape == (n,)

    Xd = rng.rand(n, 3)
    yd = (rng.rand(n) < 0.5).astype(float)
    yd[:5] = np.nan

    detection = HistogramBinning(detection=True)
    detection.fit(Xd, yd)

    assert detection.support_.tolist() == [35, 35]
    assert detection.transform(Xd).shape == (n,)


# ---------------------------------------------------------------------------
# metrics: same observation mask decides every per-class denominator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric_cls", [ECE, ACE, MCE])
def test_metric_per_class_denominator_uses_mask(metric_cls):
    n, k = 400, 3
    rng = np.random.RandomState(5)
    X = rng.rand(n, k)

    y = np.full((n, k), np.nan)
    y[:150, 0] = (rng.rand(150) < X[:150, 0]).astype(float)
    y[:, 1] = (rng.rand(n) < X[:, 1]).astype(float)
    # class 2 is unobserved

    metric = metric_cls(bins=10)
    _, class_scores = metric.measure(X, y, return_class_scores=True)

    # class 0 must be scored on exactly its 150 observed samples
    single_0 = metric.measure(X[:150, 0], y[:150, 0])
    assert np.isclose(class_scores[0], single_0)

    # class 1 is fully observed
    single_1 = metric.measure(X[:, 1], y[:, 1])
    assert np.isclose(class_scores[1], single_1)

    # the unobserved class has no denominator and is reported rather than silently scored
    assert np.isnan(class_scores[2])


def test_metric_mask_and_nan_encoding_agree():
    n, k = 200, 3
    rng = np.random.RandomState(6)
    X = rng.rand(n, k)

    y = np.full((n, k), np.nan)
    y[:80, 0] = (rng.rand(80) < X[:80, 0]).astype(float)
    y[:, 1] = (rng.rand(n) < X[:, 1]).astype(float)

    metric = ECE(bins=10)
    from_nan = metric.measure(X, y)

    # zero-filling the unknown entries must not matter once the mask is supplied - the metric
    # must use the mask instead of the (wrong) zero values
    mask = np.isfinite(y)
    from_mask = metric.measure(X, np.nan_to_num(y), mask=mask)

    assert np.isclose(from_nan, from_mask)


def test_metric_zero_filling_without_mask_would_differ():
    n, k = 200, 3
    rng = np.random.RandomState(7)
    X = rng.rand(n, k)

    y = np.full((n, k), np.nan)
    y[:80, 0] = (rng.rand(80) < X[:80, 0]).astype(float)
    y[:, 1] = (rng.rand(n) < X[:, 1]).astype(float)

    metric = ECE(bins=10)
    masked = metric.measure(X, y)

    # zero-filling *without* a mask is interpreted as fully observed negative labels and
    # therefore produces a different (polluted) value - this documents why the mask matters
    zero_filled = metric.measure(X, np.nan_to_num(y))
    assert not np.isclose(masked, zero_filled)


def test_metric_support_is_reported():
    X, y = _multilabel_data()

    _, num_samples = ECE(bins=10).measure(X, y, return_num_samples=True)

    np.testing.assert_array_equal(num_samples, [240, 80, 0])


# ---------------------------------------------------------------------------
# single-label multi-class and detection: unknown labels are a row property
# ---------------------------------------------------------------------------

def test_single_label_multiclass_per_sample_mask_drops_rows():
    n, k = 150, 3
    rng = np.random.RandomState(8)
    X = rng.rand(n, k)
    y = np.eye(k)[rng.randint(0, k, n)]

    mask = np.ones(n, dtype=bool)
    mask[:20] = False

    model = HistogramBinning()
    model.fit(X, y, mask=mask)

    # a single-label sample owns exactly one label: only unobserved rows are dropped
    assert model.transform(X).shape == (n, k)
    np.testing.assert_allclose(model.transform(X).sum(axis=1), 1.0)


def test_single_label_multiclass_contradictory_per_class_mask_raises():
    n, k = 40, 3
    rng = np.random.RandomState(9)
    X = rng.rand(n, k)
    y = np.eye(k)[rng.randint(0, k, n)]

    # a single-label sample cannot have one of its classes observed and another unknown
    mask = np.tile([True, False, True], (n, 1))

    with pytest.raises(LabelObservationError):
        HistogramBinning().fit(X, y, mask=mask)


def test_detection_unknown_labels_drop_only_unobserved_samples():
    n = 60
    rng = np.random.RandomState(10)
    X = rng.rand(n, 3)
    y = (rng.rand(n) < 0.4).astype(float)
    y[:7] = np.nan

    model = HistogramBinning(detection=True)
    model.fit(X, y)

    assert model.support_.tolist() == [53, 53]
    assert model.transform(X).shape == (n,)


def test_metric_single_label_multiclass_with_unobserved_rows():
    n, k = 300, 3
    rng = np.random.RandomState(11)
    X = rng.rand(n, k)
    y = np.eye(k)[rng.randint(0, k, n)]
    y[10:20] = np.nan

    # fully unobserved rows are dropped; the remaining one-hot target keeps the classic path
    value = ECE(bins=5).measure(X, y)

    assert 0.0 <= value <= 1.0

