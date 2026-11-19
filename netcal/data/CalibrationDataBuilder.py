# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""Pair predictions and asynchronously arriving ground truth by identity, not by array position."""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "CalibrationDataBuilder",
    "AsyncCalibrationDataBuilder",
    "CalibrationDataset",
    "CalibrationData",
    "PredictionRecord",
    "LabelRecord",
    "AsyncLabelError",
    "LabelConflictError",
    "PredictionConflictError",
    "MissingTimestampError",
    "EmptyCalibrationDataError",
    "build_calibration_data",
]


class AsyncLabelError(Exception):
    """Base class for errors raised while pairing asynchronous predictions and labels."""


class LabelConflictError(AsyncLabelError, ValueError):
    """Two labels with the same sample ID, model scope and label version disagree."""


class PredictionConflictError(AsyncLabelError, ValueError):
    """Two predictions with the same sample ID and base model version disagree."""


class MissingTimestampError(AsyncLabelError, ValueError):
    """A label cutoff is configured but a label that could be used has no timestamp."""


class EmptyCalibrationDataError(AsyncLabelError, RuntimeError):
    """No prediction could be paired with a valid label."""


@dataclass(frozen=True)
class PredictionRecord:
    """A single prediction received from a base model.

    Parameters
    ----------
    sample_id : hashable
        Stable identity of the sample. This is the join key with the ground truth; it is *not*
        the position of the prediction inside a batch.
    model_version : hashable
        Version of the base model that produced the prediction. ``(sample_id, model_version)``
        identifies a prediction uniquely.
    values : np.ndarray or tuple of np.ndarray
        Raw prediction for this sample (e.g. a confidence vector, a detection feature row or
        a ``(mean, stddev)`` tuple for probabilistic regression).
    timestamp : object, optional
        Time at which the prediction was produced/received.
    """

    sample_id: Any
    model_version: Any
    values: Any
    timestamp: Any = None


@dataclass(frozen=True)
class LabelRecord:
    """A single ground-truth label delivery (a revision of a label).

    Parameters
    ----------
    sample_id : hashable
        Identity of the sample this label belongs to.
    label_version : hashable, comparable
        Revision of the label. Higher versions supersede lower ones.
    value : np.ndarray
        Ground truth value for this sample (scalar, class label or one-hot row).
    timestamp : object, optional
        Time at which this label was produced. Used to decide whether the label is valid at the
        declared label cutoff.
    model_version : hashable, optional
        If given, the label only applies to predictions of this base model version. If ``None``,
        the label applies to all base model versions of the sample (unless a model-specific
        label exists for the examined prediction).
    """

    sample_id: Any
    label_version: Any
    value: Any
    timestamp: Any = None
    model_version: Any = None


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _normalize_value(value: Any, what: str) -> Tuple[str, Tuple[np.ndarray, ...]]:
    """Bring an array-like or tuple-of-arrays value into a canonical representation."""

    if value is None:
        raise ValueError("Parameter '%s' must not be None." % what)

    if isinstance(value, tuple):
        if len(value) == 0:
            raise ValueError("Parameter '%s' must not be an empty tuple." % what)

        parts = []
        for part in value:
            array = np.asarray(part)
            if array.dtype == object:
                raise ValueError("Parameter '%s' contains a non-numeric tuple entry." % what)
            parts.append(array)
        return "tuple", tuple(parts)

    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError("Parameter '%s' must be an array-like of numbers, not a ragged/object array." % what)
    return "array", (array,)


def _values_equal(first: Tuple[str, Tuple[np.ndarray, ...]],
                  second: Tuple[str, Tuple[np.ndarray, ...]]) -> bool:
    if first[0] != second[0] or len(first[1]) != len(second[1]):
        return False

    for a, b in zip(first[1], second[1]):
        if a.shape != b.shape:
            return False
        try:
            # treat re-delivered NaN entries as identical instead of as a conflict
            equal = np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        except TypeError:  # dtype does not support NaN comparison
            equal = np.array_equal(a, b)
        if not equal:
            return False
    return True


def _stack_values(values: Sequence[Tuple[str, Tuple[np.ndarray, ...]]],
                  what: str) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Stack per-sample values to the array format expected by the netcal calibration methods."""

    kinds = {kind for kind, _ in values}
    if len(kinds) != 1:
        raise ValueError("All %s entries must have the same structure (array or tuple of arrays)." % what)

    kind = kinds.pop()
    num_parts = {len(parts) for _, parts in values}
    if len(num_parts) != 1:
        raise ValueError("All tuple-valued %s entries must have the same number of parts." % what)

    num_parts = num_parts.pop()
    stacked = []
    for part_idx in range(num_parts):
        parts = [entry[1][part_idx] for entry in values]
        shapes = {part.shape for part in parts}
        if len(shapes) != 1:
            raise ValueError("All %s entries must have the same shape; got shapes %s." % (what, sorted(shapes)))
        stacked.append(np.stack(parts))

    if kind == "tuple":
        return tuple(stacked)
    return stacked[0]


class CalibrationDataset:
    """Immutable view of one consistent prediction/label pairing.

    Only predictions that have at least one valid label enter ``X`` and ``y``. Predictions that
    are still waiting for a label are kept in :attr:`pending` and are deliberately *never*
    zero-filled into the training data.

    Attributes
    ----------
    X : np.ndarray or tuple of np.ndarray
        Paired predictions with shape ``(n_matched, ...)`` in the format expected by the netcal
        ``fit`` methods. If the input predictions were tuples (e.g. mean/std for probabilistic
        regression), ``X`` is a tuple of arrays.
    y : np.ndarray
        Paired ground truth with shape ``(n_matched, ...)``.
    sample_ids : tuple
        Sample ID of each matched row, aligned with ``X``/``y``.
    model_versions : tuple
        Base model version of each matched row, aligned with ``X``/``y``.
    label_versions : tuple
        Version of the label that was actually used for each matched row. This records the
        provenance of the calibration data.
    label_timestamps : tuple
        Timestamp of the label that was actually used for each matched row.
    weights : np.ndarray
        Weight of each matched row. Every unique ``(sample_id, model_version)`` pair has weight
        ``1.0``; duplicated deliveries never increase it.
    pending : tuple of PredictionRecord
        Predictions without a valid label at the configured cutoff.
    late_labels : tuple of LabelRecord
        Labels whose sample has a prediction but which arrived after the configured cutoff.
    superseded_labels : tuple of LabelRecord
        Labels that were in scope of a prediction but were not selected: older revisions of the
        chosen label, revisions that were overridden by a model-specific label, etc. They are not
        counted a second time, so a revised label never trains the same sample twice.
    unpaired_labels : tuple of LabelRecord
        Labels for which no prediction exists in their model-version scope (e.g. labels for
        samples outside the prediction set).
    label_cutoff : object
        Declared label cutoff used for this build (``None`` if no cutoff was declared).
    """

    def __init__(self,
                 X: Union[np.ndarray, Tuple[np.ndarray, ...]],
                 y: np.ndarray,
                 sample_ids: Sequence[Any],
                 model_versions: Sequence[Any],
                 label_versions: Sequence[Any],
                 label_timestamps: Sequence[Any],
                 weights: np.ndarray,
                 pending: Sequence[PredictionRecord],
                 late_labels: Sequence[LabelRecord],
                 superseded_labels: Sequence[LabelRecord],
                 unpaired_labels: Sequence[LabelRecord],
                 label_cutoff: Any,
                 matched_predictions: Sequence[PredictionRecord] = (),
                 matched_labels: Sequence[LabelRecord] = ()):

        self.X = X
        self.y = y
        self.sample_ids = tuple(sample_ids)
        self.model_versions = tuple(model_versions)
        self.label_versions = tuple(label_versions)
        self.label_timestamps = tuple(label_timestamps)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.pending = tuple(pending)
        self.late_labels = tuple(late_labels)
        self.superseded_labels = tuple(superseded_labels)
        self.unpaired_labels = tuple(unpaired_labels)
        self.label_cutoff = label_cutoff

        self._matched_predictions = tuple(matched_predictions)
        self._matched_labels = tuple(matched_labels)

    @property
    def predictions(self) -> Tuple[PredictionRecord, ...]:
        """Predictions behind each matched row of ``X``, in row order."""

        return self._matched_predictions

    @property
    def labels(self) -> Tuple[LabelRecord, ...]:
        """Labels behind each matched row of ``y``, in row order."""

        return self._matched_labels

    @property
    def n_matched(self) -> int:
        """Number of predictions that were paired with a valid label."""

        return int(self.weights.size)

    @property
    def n_pending(self) -> int:
        """Number of predictions that are still waiting for a valid label."""

        return len(self.pending)

    @property
    def has_samples(self) -> bool:
        """Whether at least one prediction could be paired with a label."""

        return self.n_matched > 0

    @property
    def pending_sample_ids(self) -> Tuple[Any, ...]:
        """Sample IDs of all predictions that remain in the pending population."""

        return tuple(record.sample_id for record in self.pending)

    @property
    def pending_model_versions(self) -> Tuple[Any, ...]:
        """Base model versions of all predictions that remain in the pending population."""

        return tuple(record.model_version for record in self.pending)

    def to_arrays(self) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray]:
        """Return the paired ``(X, y)`` arrays for the existing netcal ``fit``/metric APIs.

        Unlabeled predictions are not included and no synthetic/zero labels are added.
        """

        return self.X, self.y

    def matched_records(self) -> Tuple[Tuple[PredictionRecord, LabelRecord], ...]:
        """Return the ``(prediction, label)`` pair behind each row of ``X``/``y``."""

        return tuple(zip(self._matched_predictions, self._matched_labels))

    def provenance(self) -> List[Dict[str, Any]]:
        """Row-aligned provenance: sample ID, base model version and used label version.

        This is the record to persist next to a fitted calibration model so that it is auditable
        which label revisions produced the calibration data.
        """

        return [
            {
                "row": row,
                "sample_id": sample_id,
                "model_version": model_version,
                "label_version": label_version,
                "label_timestamp": timestamp,
            }
            for row, (sample_id, model_version, label_version, timestamp) in enumerate(zip(
                self.sample_ids, self.model_versions, self.label_versions, self.label_timestamps))
        ]

    def iter_matched(self):
        """Iterate over paired rows as ``(sample_id, model_version, x, y)``."""

        for row in range(self.n_matched):
            x = self.X[row] if not isinstance(self.X, tuple) else tuple(part[row] for part in self.X)
            yield self.sample_ids[row], self.model_versions[row], x, self.y[row]

    def pending_to_arrays(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Return the pending predictions as arrays for inspection (never used for training)."""

        if len(self.pending) == 0:
            return np.empty((0, ))

        return _stack_values([_normalize_value(record.values, "pending prediction")
                              for record in self.pending], "pending predictions")

    def __len__(self) -> int:
        return self.n_matched

    def __repr__(self) -> str:
        return ("CalibrationDataset(n_matched=%d, n_pending=%d, label_cutoff=%r, "
                "model_versions=%r)" % (self.n_matched, self.n_pending,
                                        self.label_cutoff, sorted(set(self.model_versions), key=str)))


_UNSET = object()


class CalibrationDataBuilder:
    """Construct calibration data from asynchronously arriving predictions and labels.

    Concatenating predictions and labels by their position in an array silently pairs the wrong
    samples as soon as one stream lags behind the other. This builder instead keys predictions by
    ``(sample_id, model_version)`` and labels by ``(sample_id, label_version)`` and only joins
    records that belong to the same sample.

    The pairing follows these rules:

    - Labels are revisions. Among all labels valid at the declared ``label_cutoff`` (i.e. labels
      with ``timestamp <= label_cutoff``), the one with the highest ``label_version`` is used.
      A newer revision therefore revokes the pairing with the previous revision; the same sample
      is never trained twice with old and new labels.
    - Two deliveries with the same sample ID, model scope and label version but different values
      are a conflict and raise :class:`LabelConflictError`. Identical re-deliveries are ignored.
    - A prediction is identified by ``(sample_id, model_version)``. Identical re-deliveries are
      ignored and never increase the sample weight; differing values under the same identity
      raise :class:`PredictionConflictError`.
    - Predictions without a valid label remain in the pending population. They are never filled
      with zero labels and are never passed to ``fit`` or to a metric.
    - A label with ``model_version`` set only applies to predictions of that base model version.
      If such a model-specific label exists, it takes precedence over sample-wide labels
      (``model_version=None``) for that prediction.

    Parameters
    ----------
    label_cutoff : object, optional, default: None
        Declared deadline for labels used to build the calibration data. Labels with a timestamp
        after the cutoff are kept as ``late_labels`` but are not used. If ``None``, every label
        is eligible. A cutoff can also be supplied per :meth:`build`, :meth:`fit`,
        :meth:`measure` or :meth:`to_arrays` call.

    Examples
    --------
    >>> builder = CalibrationDataBuilder(label_cutoff=100)
    >>> builder.add_prediction("frame-17", "resnet-v3", confidences_a)
    >>> builder.add_prediction("frame-42", "resnet-v3", confidences_b)
    >>> # labels may arrive in a different order and may be revised
    >>> builder.add_label("frame-42", label_version=1, label=0, timestamp=90)
    >>> builder.add_label("frame-17", label_version=1, label=3, timestamp=95)
    >>> builder.add_label("frame-17", label_version=2, label=4, timestamp=120)  # too late
    >>> dataset = builder.build()
    >>> X, y = dataset.to_arrays()  # only frame-17 and frame-42 are paired; frame-17 uses v1
    """

    def __init__(self, label_cutoff: Any = None):
        self.label_cutoff = label_cutoff

        # keys: (sample_id, model_version) / (sample_id, model_version_scope, label_version)
        self._predictions: Dict[Tuple[Any, Any], PredictionRecord] = {}
        self._labels: Dict[Tuple[Any, Any, Any], LabelRecord] = {}

        self._duplicate_predictions = 0
        self._duplicate_labels = 0
        self._last_dataset: Optional[CalibrationDataset] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # adding prediction records

    def add_prediction(self, sample_id: Any, model_version: Any, prediction: Any,
                       timestamp: Any = None) -> PredictionRecord:
        """Add a single prediction, identified by sample ID and base model version.

        Parameters
        ----------
        sample_id : hashable
            Stable sample identity (never the array position within a batch).
        model_version : hashable
            Version of the base model that produced this prediction.
        prediction : array-like or tuple of array-like
            Prediction for this sample. Use a tuple for multi-part predictions such as the
            ``(mean, stddev)`` pair of probabilistic regression.
        timestamp : object, optional
            Time at which the prediction was produced/received.

        Returns
        -------
        PredictionRecord
            The stored record (the previously stored record on duplicate delivery).
        """

        if sample_id is None:
            raise ValueError("Parameter 'sample_id' must not be None: predictions are joined by ID.")
        if not _is_hashable(sample_id):
            raise ValueError("Parameter 'sample_id' must be hashable.")
        if not _is_hashable(model_version):
            raise ValueError("Parameter 'model_version' must be hashable.")

        normalized = _normalize_value(prediction, "prediction")
        key = (sample_id, model_version)

        with self._lock:
            record = self._predictions.get(key)
            if record is None:
                record = PredictionRecord(sample_id=sample_id, model_version=model_version,
                                          values=prediction, timestamp=timestamp)
                self._predictions[key] = record
                self._last_dataset = None
                return record

            stored = _normalize_value(record.values, "prediction")
            if not _values_equal(stored, normalized):
                raise PredictionConflictError(
                    "Conflicting predictions for sample ID %r and base model version %r. "
                    "Use a new base model version for corrected predictions." % (sample_id, model_version))

            # identical re-delivery: do not add another row / do not increase the weight
            self._duplicate_predictions += 1
            return record

    def add_predictions(self, sample_ids: Sequence[Any], model_versions: Sequence[Any],
                        predictions: Sequence[Any], timestamps: Optional[Sequence[Any]] = None
                        ) -> List[PredictionRecord]:
        """Add several predictions at once.

        Parameters are aligned by index and validated to have equal length.
        """

        sample_ids, model_versions, predictions, timestamps = self._check_batch(
            "predictions", sample_ids, model_versions, predictions, timestamps)

        return [self.add_prediction(sample_id, model_version, prediction, timestamp)
                for sample_id, model_version, prediction, timestamp
                in zip(sample_ids, model_versions, predictions, timestamps)]

    # convenience aliases: all of them go through the same duplicate/conflict handling
    submit_prediction = add_prediction
    push_prediction = add_prediction
    register_prediction = add_prediction

    def add_prediction_record(self, record: PredictionRecord) -> PredictionRecord:
        """Add a :class:`PredictionRecord` (e.g. one received from a prediction stream)."""

        if not isinstance(record, PredictionRecord):
            raise TypeError("Parameter 'record' must be a PredictionRecord.")

        return self.add_prediction(record.sample_id, record.model_version, record.values,
                                   timestamp=record.timestamp)

    def remove_prediction(self, sample_id: Any, model_version: Any) -> bool:
        """Remove a stored prediction. Returns ``True`` if it existed.

        The paired data is rebuilt on the next :meth:`build`, so the corresponding label - if
        any - moves to ``unpaired_labels`` instead of being silently paired with another row.
        """

        with self._lock:
            removed = self._predictions.pop((sample_id, model_version), None)
            if removed is not None:
                self._last_dataset = None
            return removed is not None

    # ------------------------------------------------------------------
    # adding label records

    def add_label(self, sample_id: Any, label_version: Any, label: Any,
                  timestamp: Any = None, model_version: Any = None) -> LabelRecord:
        """Add or revise the ground truth of a sample.

        Parameters
        ----------
        sample_id : hashable
            Identity of the sample this label belongs to.
        label_version : hashable and comparable
            Revision of the label. A higher version supersedes lower versions once it is valid
            at the label cutoff.
        label : array-like
            Ground truth for the sample.
        timestamp : object, optional
            Time at which the label was produced. Required if a label cutoff is declared.
        model_version : hashable, optional
            Restrict the label to predictions of this base model version. ``None`` means the
            label applies to every base model version of the sample.

        Raises
        ------
        LabelConflictError
            If the same sample, model scope and label version was already delivered with a
            different value.
        """

        if sample_id is None:
            raise ValueError("Parameter 'sample_id' must not be None: labels are joined by ID.")
        if not _is_hashable(sample_id):
            raise ValueError("Parameter 'sample_id' must be hashable.")
        if label_version is None:
            raise ValueError("Parameter 'label_version' must not be None.")
        if not _is_hashable(label_version):
            raise ValueError("Parameter 'label_version' must be hashable.")
        if not _is_hashable(model_version):
            raise ValueError("Parameter 'model_version' must be hashable.")

        if isinstance(label, tuple):
            raise ValueError("Parameter 'label' must be a single array-like, not a tuple.")

        normalized = _normalize_value(label, "label")
        if normalized[0] != "array":
            raise ValueError("Parameter 'label' must be a single array-like value.")
        if normalized[1][0].ndim > 2:
            raise ValueError("Parameter 'label' must have at most 2 dimensions (label vector or one-hot row).")

        key = (sample_id, model_version, label_version)

        with self._lock:
            record = self._labels.get(key)
            if record is None:
                record = LabelRecord(sample_id=sample_id, label_version=label_version,
                                     value=label, timestamp=timestamp, model_version=model_version)
                self._labels[key] = record
                self._last_dataset = None
                return record

            stored = _normalize_value(record.value, "label")
            if not _values_equal(stored, normalized):
                raise LabelConflictError(
                    "Conflicting labels for sample ID %r, base model version scope %r and label version %r. "
                    "Deliver a revised label with a new label_version instead." %
                    (sample_id, model_version, label_version))

            # identical re-delivery: ignore it so that it cannot increase the sample weight
            self._duplicate_labels += 1
            return record

    def add_labels(self, sample_ids: Sequence[Any], label_versions: Sequence[Any],
                   labels: Sequence[Any], timestamps: Optional[Sequence[Any]] = None,
                   model_versions: Union[Any, Sequence[Any], None] = None) -> List[LabelRecord]:
        """Add several labels at once.

        ``model_versions`` may be a single value (applied to all labels), ``None`` (sample-wide
        labels) or a sequence aligned with the other parameters.
        """

        sample_ids = list(sample_ids)
        label_versions = list(label_versions)
        labels = list(labels)
        length = len(sample_ids)

        if len(label_versions) != length or len(labels) != length:
            raise ValueError("Parameters 'sample_ids', 'label_versions' and 'labels' must have equal length.")

        if timestamps is None:
            timestamps = [None] * length
        else:
            timestamps = list(timestamps)
            if len(timestamps) != length:
                raise ValueError("Parameter 'timestamps' must have the same length as 'sample_ids'.")

        if model_versions is None or isinstance(model_versions, (str, bytes)) or not isinstance(model_versions, (list, tuple, np.ndarray)):
            model_versions = [model_versions] * length
        else:
            model_versions = list(model_versions)
            if len(model_versions) != length:
                raise ValueError("Parameter 'model_versions' must have the same length as 'sample_ids'.")

        return [self.add_label(sample_id, label_version, label, timestamp, model_version)
                for sample_id, label_version, label, timestamp, model_version
                in zip(sample_ids, label_versions, labels, timestamps, model_versions)]

    def add_labels_scoped(self, sample_ids: Sequence[Any], label_versions: Sequence[Any],
                          labels: Sequence[Any], model_versions: Sequence[Any],
                          timestamps: Optional[Sequence[Any]] = None) -> List[LabelRecord]:
        """Add labels that are each restricted to the base model version given in ``model_versions``.

        This is a thin, explicit wrapper around :meth:`add_labels` for callers that always have a
        per-label model version and do not want to rely on the "scalar means all" rule.
        """

        model_versions = list(model_versions)
        if len(model_versions) != len(sample_ids):
            raise ValueError("Parameter 'model_versions' must have the same length as 'sample_ids'.")

        return self.add_labels(sample_ids, label_versions, labels,
                               timestamps=timestamps, model_versions=model_versions)

    def revise_label(self, sample_id: Any, label_version: Any, label: Any,
                     timestamp: Any = None, model_version: Any = None) -> LabelRecord:
        """Alias for :meth:`add_label`, emphasizing that a label is revised."""

        return self.add_label(sample_id, label_version, label, timestamp=timestamp,
                              model_version=model_version)

    # convenience aliases
    submit_label = add_label
    push_label = add_label
    register_label = add_label

    def add_label_record(self, record: LabelRecord) -> LabelRecord:
        """Add a :class:`LabelRecord` (e.g. one received from a label stream)."""

        if not isinstance(record, LabelRecord):
            raise TypeError("Parameter 'record' must be a LabelRecord.")

        return self.add_label(record.sample_id, record.label_version, record.value,
                              timestamp=record.timestamp, model_version=record.model_version)

    def revoke_label(self, sample_id: Any, label_version: Any = None,
                     model_version: Any = _UNSET) -> int:
        """Retract stored labels so that the affected predictions fall back to older revisions
        or into the pending population.

        Parameters
        ----------
        sample_id : hashable
            Sample whose labels should be retracted.
        label_version : hashable, optional
            Specific revision to retract. If ``None``, all revisions of the sample are retracted.
        model_version : hashable, optional
            Retract only labels with this model-version scope. If omitted, all scopes are
            retracted. Pass ``model_version=None`` explicitly to retract only sample-wide labels.

        Returns
        -------
        int
            Number of label records removed.
        """

        with self._lock:
            keys = []
            for key in self._labels:
                key_sample, key_model_version, key_label_version = key
                if key_sample != sample_id:
                    continue
                if model_version is not _UNSET and key_model_version != model_version:
                    continue
                if label_version is not None and key_label_version != label_version:
                    continue
                keys.append(key)

            for key in keys:
                del self._labels[key]

            if keys:
                self._last_dataset = None

            return len(keys)

    # ------------------------------------------------------------------
    # building paired data

    def build(self, label_cutoff: Any = _UNSET) -> CalibrationDataset:
        """Join predictions and labels into a consistent calibration dataset.

        Parameters
        ----------
        label_cutoff : object, optional
            Override the cutoff declared in the constructor for this build.

        Returns
        -------
        CalibrationDataset
            Paired arrays plus the pending, late and unpaired populations and the label versions
            that were used.

        Raises
        ------
        MissingTimestampError
            If a cutoff is active and a label that could be paired has no timestamp.
        """

        cutoff = self.label_cutoff if label_cutoff is _UNSET else label_cutoff

        with self._lock:
            predictions = list(self._predictions.values())
            labels = list(self._labels.values())

        matched_predictions: List[PredictionRecord] = []
        matched_labels: List[LabelRecord] = []
        pending: List[PredictionRecord] = []

        # audit state per label: in scope of a prediction, actually selected, or too late
        labels_by_sample: Dict[Any, List[LabelRecord]] = {}
        for label in labels:
            labels_by_sample.setdefault(label.sample_id, []).append(label)

        label_keys = {self._label_key(label): label for label in labels}
        in_scope = set()
        selected = set()
        late = set()

        for prediction in predictions:
            candidates = [label for label in labels_by_sample.get(prediction.sample_id, [])
                          if label.model_version in (None, prediction.model_version)]

            if len(candidates) == 0:
                pending.append(prediction)
                continue

            # a label scoped to the base model takes precedence over sample-wide labels;
            # only if no model-specific label is valid at the cutoff, fall back to sample-wide
            specific = [label for label in candidates if label.model_version is not None]
            sample_wide = [label for label in candidates if label.model_version is None]

            for group in (specific, sample_wide):
                if len(group) == 0:
                    continue

                valid: List[LabelRecord] = []
                for label in group:
                    key = self._label_key(label)
                    in_scope.add(key)

                    if cutoff is not None:
                        if label.timestamp is None:
                            raise MissingTimestampError(
                                "Label (sample ID %r, label version %r) has no timestamp but a label cutoff %r is "
                                "declared. Deliver a timestamp for every label." %
                                (label.sample_id, label.label_version, cutoff))

                        if self._is_after(label.timestamp, cutoff):
                            late.add(key)
                            continue

                    valid.append(label)

                if len(valid) == 0:
                    continue

                try:
                    newest = max(valid, key=lambda label: label.label_version)
                except TypeError as error:
                    raise ValueError(
                        "Label versions of sample ID %r are not mutually comparable: %r." %
                        (prediction.sample_id, [label.label_version for label in valid])) from error

                selected.add(self._label_key(newest))
                matched_predictions.append(prediction)
                matched_labels.append(newest)
                break
            else:
                # every revision in scope arrived too late (or none is valid) -> stays pending
                pending.append(prediction)

        # a label is late if it arrived after the cutoff, superseded if a newer revision of the
        # same sample won at the cutoff, and unpaired if it has no prediction in its scope at all
        late_labels = []
        superseded_labels = []
        unpaired_labels = []
        for key, label in label_keys.items():
            if key in late:
                late_labels.append(label)
            elif key in selected:
                continue
            elif key in in_scope or self._model_scope_in_scope(label, predictions):
                superseded_labels.append(label)
            else:
                unpaired_labels.append(label)

        if len(matched_predictions) > 0:
            X = _stack_values([_normalize_value(record.values, "prediction")
                               for record in matched_predictions], "matched predictions")
            y = _stack_values([_normalize_value(record.value, "label")
                               for record in matched_labels], "matched labels")
            weights = np.ones(len(matched_predictions), dtype=np.float64)
        else:
            # no fake labels and no zero-filled rows: simply stay empty
            X = np.empty((0, ))
            y = np.empty((0, ))
            weights = np.zeros(0, dtype=np.float64)

        dataset = CalibrationDataset(
            X=X, y=y,
            sample_ids=[record.sample_id for record in matched_predictions],
            model_versions=[record.model_version for record in matched_predictions],
            label_versions=[record.label_version for record in matched_labels],
            label_timestamps=[record.timestamp for record in matched_labels],
            weights=weights,
            pending=pending,
            late_labels=late_labels,
            superseded_labels=superseded_labels,
            unpaired_labels=unpaired_labels,
            label_cutoff=cutoff,
            matched_predictions=matched_predictions,
            matched_labels=matched_labels,
        )

        with self._lock:
            self._last_dataset = dataset

        return dataset

    def to_arrays(self, label_cutoff: Any = _UNSET
                  ) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], np.ndarray]:
        """Build the pairing and return ``(X, y)`` for the existing netcal ``fit`` methods."""

        return self.build(label_cutoff=label_cutoff).to_arrays()

    @property
    def pending(self) -> Tuple[PredictionRecord, ...]:
        """Predictions without a valid label in the most recent build."""

        return self.build().pending

    @property
    def last_dataset(self) -> Optional[CalibrationDataset]:
        """Dataset returned by the most recent :meth:`build`, :meth:`fit` or :meth:`measure`."""

        return self._last_dataset

    @property
    def stats(self) -> Dict[str, int]:
        """Counters for auditing duplicate deliveries and the stored populations."""

        with self._lock:
            return {
                "predictions": len(self._predictions),
                "labels": len(self._labels),
                "duplicate_predictions": self._duplicate_predictions,
                "duplicate_labels": self._duplicate_labels,
            }

    # ------------------------------------------------------------------
    # feeding the existing fit/metric APIs

    def fit(self, calibrator, label_cutoff: Any = _UNSET, **fit_kwargs):
        """Fit an existing netcal calibration method on the paired real ``(X, y)`` data.

        Only rows with an actually arrived, valid label are used. The label versions that were
        used are stored on the fitted calibrator as ``label_versions_`` (and the dataset as
        ``calibration_dataset_``) so that the calibration is reproducible and auditable.

        Parameters
        ----------
        calibrator : instance of a netcal calibration method
            Any object exposing ``fit(X, y, **kwargs)``, e.g.
            :class:`netcal.binning.HistogramBinning` or
            :class:`netcal.scaling.TemperatureScaling`.
        label_cutoff : object, optional
            Override the declared cutoff for this build.

        Returns
        -------
        calibrator
            The fitted calibration method.

        Raises
        ------
        EmptyCalibrationDataError
            If no prediction has a valid label. Training on synthetic/zero labels is never used
            as a fallback.
        """

        dataset = self.build(label_cutoff=label_cutoff)
        if not dataset.has_samples:
            raise EmptyCalibrationDataError(
                "No prediction has a valid label at label cutoff %r: %d prediction(s) are pending. "
                "Refusing to fit on labels that have not arrived." %
                (dataset.label_cutoff, dataset.n_pending))

        calibrator.fit(dataset.X, dataset.y, **fit_kwargs)

        calibrator.label_versions_ = dataset.label_versions
        calibrator.calibration_dataset_ = dataset
        return calibrator

    def measure(self, metric, label_cutoff: Any = _UNSET, **measure_kwargs):
        """Evaluate an existing netcal metric on the paired real ``(X, y)`` data.

        Parameters
        ----------
        metric : instance of a netcal metric
            Any object exposing ``measure(X, y, **kwargs)``, e.g. :class:`netcal.metrics.ECE`.
        label_cutoff : object, optional
            Override the declared cutoff for this build.
        **measure_kwargs
            Forwarded to ``metric.measure``.

        Returns
        -------
        object
            Whatever the metric returns.

        Raises
        ------
        EmptyCalibrationDataError
            If no prediction has a valid label.
        """

        dataset = self.build(label_cutoff=label_cutoff)
        if not dataset.has_samples:
            raise EmptyCalibrationDataError(
                "No prediction has a valid label at label cutoff %r: %d prediction(s) are pending. "
                "Refusing to evaluate on labels that have not arrived." %
                (dataset.label_cutoff, dataset.n_pending))

        self._last_dataset = dataset
        return metric.measure(dataset.X, dataset.y, **measure_kwargs)

    # alias: "evaluate" reads naturally for error evaluation
    evaluate = measure

    # ------------------------------------------------------------------
    # helpers

    @staticmethod
    def _model_scope_in_scope(label: LabelRecord, predictions: Sequence[PredictionRecord]) -> bool:
        """A model-specific label is in scope only if a prediction of that base model exists."""

        if label.model_version is None:
            return any(prediction.sample_id == label.sample_id for prediction in predictions)
        return any(prediction.sample_id == label.sample_id and prediction.model_version == label.model_version
                   for prediction in predictions)

    @staticmethod
    def _label_key(label: LabelRecord) -> Tuple[Any, Any, Any]:
        return label.sample_id, label.model_version, label.label_version

    @staticmethod
    def _is_after(timestamp: Any, cutoff: Any) -> bool:
        try:
            return bool(timestamp > cutoff)
        except TypeError as error:
            raise TypeError(
                "Label timestamps and label cutoff must be comparable (got timestamp %r and cutoff %r)." %
                (timestamp, cutoff)) from error

    @staticmethod
    def _check_batch(what: str, sample_ids: Sequence[Any], model_versions: Sequence[Any],
                     values: Sequence[Any], timestamps: Optional[Sequence[Any]]):
        sample_ids = list(sample_ids)
        model_versions = list(model_versions)
        values = list(values)

        if not (len(sample_ids) == len(model_versions) == len(values)):
            raise ValueError("Parameters 'sample_ids', 'model_versions' and '%s' must have equal length." % what)

        if timestamps is None:
            timestamps = [None] * len(sample_ids)
        else:
            timestamps = list(timestamps)
            if len(timestamps) != len(sample_ids):
                raise ValueError("Parameter 'timestamps' must have the same length as 'sample_ids'.")

        return sample_ids, model_versions, values, timestamps

    def __len__(self) -> int:
        with self._lock:
            return len(self._predictions)

    def __repr__(self) -> str:
        stats = self.stats
        return ("CalibrationDataBuilder(label_cutoff=%r, predictions=%d, labels=%d, "
                "duplicate_predictions=%d, duplicate_labels=%d)" %
                (self.label_cutoff, stats["predictions"], stats["labels"],
                 stats["duplicate_predictions"], stats["duplicate_labels"]))


# descriptive alias for discoverability
AsyncCalibrationDataBuilder = CalibrationDataBuilder

# alias for the paired-data container
CalibrationData = CalibrationDataset


def build_calibration_data(predictions: Sequence[PredictionRecord],
                           labels: Sequence[LabelRecord],
                           label_cutoff: Any = None) -> CalibrationDataset:
    """Build a :class:`CalibrationDataset` from already-materialized records.

    This is a convenience wrapper around :class:`CalibrationDataBuilder` for callers that have
    collected records first (e.g. from a message queue or a database) and only want to construct
    the paired arrays.

    Parameters
    ----------
    predictions : iterable of PredictionRecord
        Predictions to pair.
    labels : iterable of LabelRecord
        Label deliveries (all revisions) to pair with the predictions.
    label_cutoff : object, optional
        Declared label cutoff; see :class:`CalibrationDataBuilder`.

    Returns
    -------
    CalibrationDataset
        The paired dataset, including pending predictions and provenance information.
    """

    builder = CalibrationDataBuilder(label_cutoff=label_cutoff)

    for record in predictions:
        builder.add_prediction_record(record)

    for record in labels:
        builder.add_label_record(record)

    return builder.build()
