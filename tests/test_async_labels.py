"""Tests for pairing asynchronously arriving predictions and labels (netcal.data)."""

import numpy as np
import pytest

from netcal.data import (
    CalibrationDataBuilder,
    EmptyCalibrationDataError,
    LabelConflictError,
    MissingTimestampError,
    PredictionConflictError,
)


def confidence(sample):
    """Deterministic prediction so that mis-pairing is visible in the values."""
    return np.array([1.0 - sample / 10.0, sample / 10.0])


def test_positional_pairing_is_avoided():
    # labels arrive in a different order than the predictions
    builder = CalibrationDataBuilder()
    builder.add_predictions(["a", "b", "c"], ["v1"] * 3,
                            [confidence(1), confidence(5), confidence(9)])
    builder.add_labels(["c", "a", "b"], [1, 1, 1], [9, 1, 5])

    dataset = builder.build()

    assert dataset.sample_ids == ("a", "b", "c")
    np.testing.assert_allclose(dataset.X, np.stack([confidence(1), confidence(5), confidence(9)]))
    np.testing.assert_array_equal(dataset.y, np.array([1, 5, 9]))


def test_labels_are_keyed_by_sample_id_and_model_version():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(2))
    builder.add_prediction("a", "v2", confidence(4))
    builder.add_label("a", 1, 2, model_version="v1")

    dataset = builder.build()

    # only the v1 prediction gets its model-specific label; the v2 prediction stays pending
    assert dataset.model_versions == ("v1",)
    assert dataset.pending_model_versions == ("v2",)
    np.testing.assert_array_equal(dataset.y, np.array([2]))


def test_latest_valid_label_at_cutoff_is_selected():
    builder = CalibrationDataBuilder(label_cutoff=100)
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", label_version=1, label=3, timestamp=90)
    builder.add_label("a", label_version=2, label=4, timestamp=101)  # too late

    dataset = builder.build()

    assert dataset.label_versions == (1, )
    np.testing.assert_array_equal(dataset.y, np.array([3]))
    assert [record.label_version for record in dataset.late_labels] == [2]


def test_newer_revision_within_cutoff_revokes_old_pairing():
    builder = CalibrationDataBuilder(label_cutoff=100)
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", label_version=1, label=3, timestamp=90)
    builder.add_label("a", label_version=2, label=7, timestamp=95)

    dataset = builder.build()

    # one row only, with the revised label - the old revision does not double-count the sample
    assert dataset.n_matched == 1
    assert dataset.label_versions == (2, )
    np.testing.assert_array_equal(dataset.y, np.array([7]))
    assert [record.label_version for record in dataset.superseded_labels] == [1]


def test_revocation_moves_pairing_back_to_the_previous_revision():
    builder = CalibrationDataBuilder(label_cutoff=100)
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", label_version=1, label=3, timestamp=90)
    builder.add_label("a", label_version=2, label=7, timestamp=95)

    assert builder.revoke_label("a", label_version=2) == 1

    dataset = builder.build()
    np.testing.assert_array_equal(dataset.y, np.array([3]))
    assert dataset.label_versions == (1, )


def test_revocation_of_all_labels_moves_prediction_to_pending():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", 1, 3)

    builder.revoke_label("a")

    dataset = builder.build()
    assert not dataset.has_samples
    assert dataset.pending_sample_ids == ("a",)


def test_conflicting_labels_for_same_version_are_rejected():
    builder = CalibrationDataBuilder()
    builder.add_label("a", label_version=1, label=3)

    with pytest.raises(LabelConflictError):
        builder.add_label("a", label_version=1, label=4)

    # same value re-delivered is not a conflict, just an ignored duplicate
    builder.add_label("a", label_version=1, label=3)
    assert builder.stats["duplicate_labels"] == 1


def test_conflicting_predictions_for_same_model_version_are_rejected():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))

    with pytest.raises(PredictionConflictError):
        builder.add_prediction("a", "v1", confidence(4))


def test_duplicate_deliveries_do_not_increase_weight():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", 1, 3)
    builder.add_label("a", 1, 3)

    dataset = builder.build()

    assert dataset.n_matched == 1
    np.testing.assert_array_equal(dataset.weights, np.array([1.0]))
    assert builder.stats["duplicate_predictions"] == 2
    assert builder.stats["duplicate_labels"] == 1


def test_unlabeled_predictions_stay_pending_and_are_not_zero_filled():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_prediction("b", "v1", confidence(6))
    builder.add_label("a", 1, 3)

    dataset = builder.build()
    X, y = dataset.to_arrays()

    assert X.shape == (1, 2)
    np.testing.assert_array_equal(y, np.array([3]))
    assert dataset.pending_sample_ids == ("b",)
    assert dataset.n_pending == 1


def test_labels_without_predictions_are_reported_separately():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", 1, 3)
    builder.add_label("z", 1, 5)

    dataset = builder.build()

    assert [record.sample_id for record in dataset.unpaired_labels] == ["z"]


def test_missing_timestamp_with_active_cutoff_is_rejected():
    builder = CalibrationDataBuilder(label_cutoff=100)
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", 1, 3)

    with pytest.raises(MissingTimestampError):
        builder.build()


def test_cutoff_can_be_supplied_per_build():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))
    builder.add_label("a", label_version=1, label=3, timestamp=50)
    builder.add_label("a", label_version=2, label=7, timestamp=150)

    early = builder.build(label_cutoff=100)
    late = builder.build(label_cutoff=200)

    assert early.label_versions == (1,)
    assert late.label_versions == (2,)


def test_fit_uses_only_paired_real_data_and_records_label_versions():
    from netcal.binning import HistogramBinning

    builder = CalibrationDataBuilder()
    builder.add_predictions(["a", "b", "c", "d"], ["v1"] * 4,
                            [np.array([0.09]), np.array([0.48]), np.array([0.51]), np.array([0.91])])
    builder.add_labels(["a", "b", "c"], [1, 1, 1], [0, 1, 0])  # "d" has no label yet

    dataset = builder.build()
    assert dataset.n_pending == 1
    assert dataset.pending_sample_ids == ("d",)

    calibrator = builder.fit(HistogramBinning(bins=2))

    # the calibration seen exactly the paired rows - no zero label for "d" sneaks in
    reference = HistogramBinning(bins=2).fit(dataset.X, dataset.y)
    probe = np.linspace(0.0, 1.0, 11).reshape(-1, 1)
    np.testing.assert_allclose(calibrator.transform(probe), reference.transform(probe))

    np.testing.assert_array_equal(calibrator.label_versions_, (1, 1, 1))
    assert calibrator.calibration_dataset_.pending_sample_ids == ("d",)


def test_measure_uses_the_paired_data():
    from netcal.metrics import ECE

    builder = CalibrationDataBuilder()
    # predictions confidently wrong -> ECE 1.0 if the right labels are paired
    builder.add_prediction("a", "v1", np.array([0.0, 1.0]))
    builder.add_label("a", 1, 0)

    assert builder.measure(ECE(bins=2)) == pytest.approx(1.0)


def test_fit_refuses_when_no_label_has_arrived():
    from netcal.binning import HistogramBinning

    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", confidence(3))

    with pytest.raises(EmptyCalibrationDataError):
        builder.fit(HistogramBinning(bins=2))


def test_model_specific_label_falls_back_to_sample_wide_label_when_late():
    builder = CalibrationDataBuilder(label_cutoff=100)
    builder.add_prediction("a", "v2", confidence(4))
    builder.add_label("a", 1, 4, timestamp=101, model_version="v2")  # too late
    builder.add_label("a", 1, 4, timestamp=90)                       # sample-wide, valid

    dataset = builder.build()

    assert dataset.n_matched == 1
    np.testing.assert_array_equal(dataset.y, np.array([4]))
    assert [record.model_version for record in dataset.late_labels] == ["v2"]


def test_redundant_prediction_with_nan_is_not_a_conflict():
    builder = CalibrationDataBuilder()
    builder.add_prediction("a", "v1", np.array([0.2, np.nan]))
    builder.add_prediction("a", "v1", np.array([0.2, np.nan]))

    assert builder.stats["duplicate_predictions"] == 1


def test_build_calibration_data_from_records():
    from netcal.data import LabelRecord, PredictionRecord, build_calibration_data

    predictions = [PredictionRecord("a", "v1", confidence(2)),
                   PredictionRecord("b", "v1", confidence(8))]
    labels = [LabelRecord("b", 1, 8, timestamp=10),
              LabelRecord("a", 1, 2, timestamp=10)]

    dataset = build_calibration_data(predictions, labels, label_cutoff=20)

    assert dataset.sample_ids == ("a", "b")
    np.testing.assert_array_equal(dataset.y, np.array([2, 8]))


def test_regression_tuple_predictions_are_stacked_for_fit():
    from netcal.regression import VarianceScaling

    builder = CalibrationDataBuilder()
    for i, sample_id in enumerate(["a", "b", "c", "d"]):
        builder.add_prediction(sample_id, "v1", (np.array([float(i)]), np.array([1.0])))

    builder.add_labels(["a", "b", "c", "d"], [1, 1, 1, 1],
                       [np.array([0.0]), np.array([1.0]), np.array([2.0]), np.array([3.0])])

    X, y = builder.build().to_arrays()
    assert isinstance(X, tuple) and X[0].shape == (4, 1)
    assert y.shape == (4, 1)

    scaling = builder.fit(VarianceScaling())
    assert scaling.transform((np.array([[0.0]]), np.array([[1.0]]))).shape == (1, 1)
