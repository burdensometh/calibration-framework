# Calibration data from asynchronous predictions and labels

In many production settings the predictions are produced and stored immediately, while the
ground truth arrives later, out of order, and may be corrected afterwards. Concatenating the
prediction array and the label array by position is wrong in that setting: as soon as one stream
lags behind the other, sample *i* of the predictions is paired with the label of a different
sample. The resulting calibration model looks plausible but is trained on noise.

`netcal.data.CalibrationDataBuilder` builds calibration data by **identity instead of position**:

- predictions are keyed by `(sample_id, model_version)`;
- labels are keyed by `(sample_id, label_version)` and joined only to predictions of the same
  sample;
- the label selected for a sample is the newest revision that is valid at the declared
  **label cutoff**, i.e. `timestamp <= label_cutoff`;
- a conflicting label for the same sample, model scope and label version is **rejected**;
- predictions without a valid label remain in the **pending population** and are never passed to
  `fit` or to a metric - they are *not* filled with zero labels;
- duplicated deliveries are idempotent and **do not increase the sample weight**;
- the label versions that were used are stored on the dataset and on the fitted calibrator.

The paired `X`/`y` are the same formats that the existing calibration methods and metrics
already accept, so no changes to the calibration methods are required:

```python
from netcal.data import CalibrationDataBuilder
from netcal.binning import HistogramBinning
from netcal.metrics import ECE

# declared deadline: labels produced after this time are not used for this calibration build
builder = CalibrationDataBuilder(label_cutoff=1_700_000_000)

# predictions and labels arrive independently, in any order
builder.add_predictions(sample_ids, model_versions, confidences)
builder.add_labels(label_sample_ids, label_versions, labels, timestamps=label_timestamps)

# pair, inspect and train on the real (X, y) only
dataset = builder.build()
print(dataset.n_matched, dataset.pending_sample_ids)

calibrator = builder.fit(HistogramBinning(bins=10))
print(calibrator.label_versions_)         # provenance of the training rows
print(calibrator.calibration_dataset_.provenance())

# measure on the very same pairing
ece = builder.measure(ECE(bins=10))
```

## Label revisions

Labels are revisions of each other, ordered by `label_version`. If a revision is delivered and
accepted, the sample is trained once with that revision; older revisions are reported in
`dataset.superseded_labels` and are not additionally counted. A revision that arrives after the
cutoff stays in `dataset.late_labels` and cannot silently change the calibration data. A label
can also be explicitly retracted:

```python
builder.revoke_label("sample-42", label_version=3)   # fall back to version 2, or to pending
```

## Model versions

`(sample_id, model_version)` identifies a prediction uniquely, so several base models can be
calibrated from the same builder without mixing their predictions. A label may be sample-wide
(`model_version=None`) or restricted to a base model version. A model-specific label takes
precedence over a sample-wide label for that prediction only.
