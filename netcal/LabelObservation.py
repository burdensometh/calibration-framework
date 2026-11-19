# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Label observation contract for independent (multi-label) calibration
====================================================================

Independent multi-label data is not the same as a multi-class problem: for a
single sample, *some* of the labels might be known while others are unknown.
An unknown label is **not** a negative label. Filling unknown entries with ``0``
(or deleting the complete sample) would silently change the calibration target
and pollute the fitted calibration mapping.

This module defines the observation-mask contract used throughout :mod:`netcal`:

* ``y`` is a 2-D array of shape ``(n_samples, n_classes)`` with binary labels
  (``0`` or ``1``). Unknown labels are represented by ``numpy.nan`` and/or by a
  boolean observation mask.
* ``mask`` is a boolean array of shape ``(n_samples, n_classes)``. ``True``
  means *the label for this sample and this class is known*; ``False`` means the
  label is unknown and must be ignored for that class only.
* An error is raised if an entry marked as observed does not contain a valid
  binary label.
* The mask is metadata only. It must never be appended to, multiplied with or
  otherwise merged into the feature matrix ``X``.

Use :func:`prepare_label_matrix` to validate user input and to obtain the
canonical label matrix as well as the canonical observation mask. Use
:func:`class_status` to determine whether a class can be fitted at all.
"""

import numpy as np
from typing import NamedTuple, Optional, Tuple


class LabelObservationError(ValueError):
    """Raised if labels and their observation mask violate the contract."""


class LabelObservation(NamedTuple):
    """
    Canonical result of :func:`prepare_label_matrix`.

    Attributes
    ----------
    labels : np.ndarray, shape=(n_samples, n_classes)
        Binary labels as float array. Unobserved entries are ``numpy.nan``.
    mask : np.ndarray, shape=(n_samples, n_classes), dtype=bool
        ``True`` where the label is observed.
    support : np.ndarray, shape=(n_classes,), dtype=int
        Number of observed labels per class.
    num_classes : int
        Number of classes (columns) in ``labels``.
    """

    labels: np.ndarray
    mask: np.ndarray
    support: np.ndarray
    num_classes: int


def _as_label_matrix(y, num_classes: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert input labels to a float matrix. Return labels and an optional masked-array mask."""

    if isinstance(y, np.ma.MaskedArray):
        # a masked entry is *unknown*. Preserve this information instead of
        # using the (arbitrary) filled value.
        y_mask = ~np.ma.getmaskarray(y)
        labels = np.asarray(y.filled(np.nan), dtype=np.float64)
    else:
        y_mask = None
        labels = np.asarray(y, dtype=np.float64)

    if labels.ndim == 1:
        if num_classes is None:
            num_classes = 2
        if num_classes != 2:
            raise LabelObservationError(
                "A 1-D label vector can only be interpreted for 2 classes. "
                "For multi-label data pass y with shape (n_samples, n_classes)."
            )

        # 1-D vector denotes the label of the positive class. The negative class
        # is its complement - both share the same observation state.
        labels = np.stack((1.0 - labels, labels), axis=1)
        if y_mask is not None:
            y_mask = np.stack((y_mask, y_mask), axis=1)

    elif labels.ndim == 2:
        if num_classes is not None and labels.shape[1] != num_classes:
            raise LabelObservationError(
                "Label matrix has %d columns but %d classes were inferred from the confidence array."
                % (labels.shape[1], num_classes)
            )
        num_classes = labels.shape[1]

    else:
        raise LabelObservationError(
            "Labels for multi-label calibration must be 1-D (binary) or 2-D (per class), got %d dimensions."
            % labels.ndim
        )

    return labels, y_mask


def prepare_label_matrix(
        y,
        mask=None,
        *,
        num_classes: Optional[int] = None
) -> LabelObservation:
    """
    Validate labels and build the canonical label/observation representation.

    Parameters
    ----------
    y : np.ndarray or np.ma.MaskedArray, shape=(n_samples, n_classes) or (n_samples,)
        Binary labels. Unknown entries may be encoded as ``numpy.nan`` or by
        masking a :class:`numpy.ma.MaskedArray`. A 1-D vector is only accepted
        for two classes and denotes the label of the positive class.
    mask : array-like of bool, shape=(n_samples, n_classes) or (n_samples,), optional
        Explicit observation mask. If given, it overrides the observation state
        inferred from ``y``. An entry marked as observed must contain a valid
        binary label.
    num_classes : int, optional
        Expected number of classes. Validated against ``y``/``mask``.

    Returns
    -------
    LabelObservation
        Canonical labels (unknown entries are NaN), observation mask and support
        per class.

    Raises
    ------
    LabelObservationError
        If shapes are inconsistent, if an observed entry is not binary or if no
        label is observed at all.
    """

    labels, masked_array_mask = _as_label_matrix(y, num_classes)

    # observation state encoded directly by the labels
    inferred = ~np.isnan(labels)
    if masked_array_mask is not None:
        inferred &= masked_array_mask

    if mask is None:
        observed = inferred

    else:
        mask = np.asarray(mask)
        if mask.ndim == 1 and labels.ndim == 2 and labels.shape[1] == 2:
            # broadcasting a per-sample mask to both binary columns is
            # unambiguous: both columns belong to the same sample.
            mask = np.stack((mask, mask), axis=1)
        elif mask.shape != labels.shape:
            raise LabelObservationError(
                "Observation mask has shape %s but labels have shape %s."
                % (str(mask.shape), str(labels.shape))
            )

        observed = mask.astype(bool)

        # An entry explicitly marked as observed must actually carry a label.
        conflicting = observed & ~inferred
        if np.any(conflicting):
            raise LabelObservationError(
                "Observation mask marks %d entries as observed but their label is unknown (NaN or masked)."
                % int(np.count_nonzero(conflicting))
            )

    if not np.any(observed):
        raise LabelObservationError("No observed labels provided.")

    # unknown labels are *not* negative samples. They are excluded per class.
    observed_values = labels[observed]
    if not np.all(np.isin(observed_values, (0.0, 1.0))):
        raise LabelObservationError(
            "Observed labels must be binary values in {0, 1}. Unknown labels must be encoded "
            "as NaN or passed through the observation mask."
        )

    labels = np.where(observed, labels, np.nan)
    support = observed.sum(axis=0).astype(int)

    return LabelObservation(labels=labels, mask=observed, support=support, num_classes=labels.shape[1])


def class_status(
        labels: np.ndarray,
        mask: np.ndarray,
        *,
        min_support: int = 1
) -> Tuple[list, np.ndarray]:
    """
    Determine whether each class can be fitted to a binary calibration method.

    A class is fittable only if at least ``min_support`` labels are observed and
    both binary outcomes occur among the observed labels.

    Parameters
    ----------
    labels : np.ndarray, shape=(n_samples, n_classes)
        Canonical label matrix as returned by :func:`prepare_label_matrix`.
    mask : np.ndarray, shape=(n_samples, n_classes), dtype=bool
        Observation mask.
    min_support : int, default: 1
        Minimum number of observed labels required to fit a class.

    Returns
    -------
    tuple of (list, np.ndarray)
        List with one status string per class and the per-class support. The
        status is one of:

        - ``'fitted'``: sufficient observed labels with both outcomes.
        - ``'unobserved'``: no label observed for this class.
        - ``'insufficient'``: fewer than ``min_support`` labels observed.
        - ``'degenerate'``: all observed labels have the same binary outcome, so
          a binary calibration mapping cannot be identified. Callers must fall
          back explicitly (for instance to the identity mapping).
    """

    support = mask.sum(axis=0).astype(int)
    statuses = []

    for k in range(labels.shape[1]):
        observed = mask[:, k]
        n_observed = int(support[k])

        if n_observed == 0:
            statuses.append('unobserved')
        elif n_observed < min_support:
            statuses.append('insufficient')
        elif np.unique(labels[observed, k]).size < 2:
            statuses.append('degenerate')
        else:
            statuses.append('fitted')

    return statuses, support
