# Scaling Methods for Confidence Calibration

This package consists of several methods for confidence calibration which use confidence scaling to approximate
confidence estimates to observed accuracy. The most common scaling methods are
*netcal.scaling.TemperatureScaling*, *netcal.scaling.LogisticCalibration*, and *netcal.scaling.BetaCalibration*.
Note that all methods can also be applied to object detection and are capable of additional influenting factors
such as object position and/or shape.
The advanced methods *netcal.scaling.LogisticCalibrationDependent* and *netcal.scaling.BetaCalibrationDependent*
are able to better represent possible correlations as the underlying probability distributions are joint
multivariate distributions with possible correlations.

## Reusable posterior sampling plans (VI / MCMC)

On the probabilistic methods `variational` and `mcmc`, `transform` draws parameter vectors from the posterior and
returns one calibrated output per draw. If the input data is processed in several chunks and each chunk is
transformed independently, every chunk creates its own random draws: draws with the same index then belong to
different parameter vectors, so the sample correlations are lost. Relying on identical random seeds for each
chunk only aligns the draws by accident of the random number generator state.

A `PosteriorSamplingPlan` separates drawing the parameter identities from applying them to data. Create the plan
once and pass it to every chunked `transform` call so that all chunks apply the identical parameter vectors;
concatenating the chunked results then equals a single transform of the full batch up to floating point
tolerance. A plan is bound to the fitting generation, inference method, parameter structure and dtype - re-fitting
or clearing the model invalidates it and any further application raises a `RuntimeError`.

```python
from netcal.scaling import LogisticCalibration

model = LogisticCalibration(method='variational')
model.fit(X_train, y_train)

# fix the posterior parameter identities once
plan = model.create_sampling_plan(num_samples=1000, random_state=0)

# reuse the very same plan for every input chunk (also across processes via pickle)
calibrated_chunks = [model.transform(chunk, sampling_plan=plan) for chunk in chunks]
calibrated = np.concatenate(calibrated_chunks, axis=1)
```