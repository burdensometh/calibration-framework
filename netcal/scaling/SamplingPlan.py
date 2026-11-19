# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

from collections import OrderedDict
from typing import Dict

import torch


class PosteriorSamplingPlan(object):
    """
    Reusable posterior sampling plan for Bayesian (Variational Inference / MCMC) logistic calibration models.

    A VI/MCMC :meth:`~netcal.scaling.AbstractLogisticRegression.transform` does not only apply the calibration
    mapping to the given input data but also draws a set of parameter vectors from the approximated posterior.
    Each output slice along the first axis corresponds to the calibration result obtained from one concrete
    parameter draw (one "model identity").

    If the input data is processed in several chunks and each chunk is transformed independently, every chunk
    creates its own random parameter draws. The draws with the same slice index are then generated from different
    parameter vectors, i.e. they do not belong to the same model identity anymore and the correlation between the
    posterior samples is lost when the chunked results are concatenated. Relying on identical random seeds for each
    chunk only makes the draws coincide by accident of the global random number generator state.

    A sampling plan resolves this issue by separating the random part (drawing the parameter identities) from the
    deterministic part (applying fixed parameter vectors to input data):

    1. Create a plan once via :meth:`~netcal.scaling.AbstractLogisticRegression.create_sampling_plan`.
       The plan materializes a fixed set of posterior parameter draws - these are the parameter identities.
    2. Pass the very same plan to every chunked
       :meth:`~netcal.scaling.AbstractLogisticRegression.transform` call. Each chunk applies the identical
       parameter vectors in the same order.
    3. Concatenate the chunked results along the data axis. The result is the same (up to floating point
       tolerance) as transforming the complete batch at once, because every output slice is obtained from the
       same parameter identity on every chunk.

    A plan is bound to the concrete fitted model state it was created from. It records the fitting generation
    (incremented on every fit/clear), the inference method, the parameter (site) structure including the data
    dtype of the fit. Re-fitting the model invalidates the plan and any attempt to apply it afterwards raises a
    :class:`RuntimeError` instead of silently returning correlated-but-wrong results. Plans are plain picklable
    Python objects and may be stored, transferred to worker processes and reused as long as the underlying model
    is not refitted.

    Parameters
    ----------
    method : str
        Inference method the plan was created for, either 'variational' or 'mcmc'.
    samples : OrderedDict of torch.Tensor
        Materialized posterior parameter draws. Each tensor is stored on CPU and has shape
        ``(num_draws, num_parameters_of_site)``, keyed by the site name.
    fit_generation : int
        Monotonic counter of the model's fitting generation at plan creation time.
    model_uid : str
        Unique identity of the model instance that created the plan. Prevents applying a plan to a
        different model even if the fitting generation happens to match.
    dtype : torch.dtype
        Data type of the fitted model for which the parameter draws were created.
    site_structure : OrderedDict of str to tuple
        Mapping of site names to their parameter shapes at plan creation time.
    random_state : int, optional
        Random seed that was used to create the draws (for reproducibility/introspection only).
    mcmc_indices : torch.Tensor, optional
        For MCMC plans, the chain sample indices that were selected from the stored MCMC draws.
    """

    def __init__(
            self,
            method: str,
            samples: 'OrderedDict[str, torch.Tensor]',
            fit_generation: int,
            model_uid: str,
            dtype: torch.dtype,
            site_structure: 'OrderedDict[str, tuple]',
            random_state: int = None,
            mcmc_indices: torch.Tensor = None
    ):
        """ Create a sampling plan. Instances are normally obtained from the model, not constructed directly. """

        if method not in ('variational', 'mcmc'):
            raise ValueError("Sampling plans are only available for methods 'variational' and 'mcmc', "
                             "but got '%s'." % method)

        if len(samples) == 0:
            raise ValueError("A sampling plan requires at least one posterior parameter site.")

        num_draws = None
        for name, tensor in samples.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Posterior draw for site '%s' must be a torch.Tensor." % name)

            if num_draws is None:
                num_draws = tensor.shape[0]
            elif tensor.shape[0] != num_draws:
                raise ValueError("All sites of a sampling plan must share the same number of draws.")

            # plans are stored on CPU so that they are independent of the inference device
            samples[name] = tensor.detach().cpu()

        if num_draws is None or num_draws <= 0:
            raise ValueError("A sampling plan requires at least one posterior parameter draw.")

        if mcmc_indices is not None and tuple(mcmc_indices.shape) != (num_draws,):
            raise ValueError("Number of MCMC indices must equal the number of posterior draws.")

        self.method = method
        self.samples = samples
        self.fit_generation = fit_generation
        self.model_uid = model_uid
        self.dtype = dtype
        self.site_structure = site_structure
        self.random_state = random_state
        self.mcmc_indices = None if mcmc_indices is None else mcmc_indices.detach().cpu()

    @property
    def num_draws(self) -> int:
        """ Number of materialized parameter draws (parameter identities) of this plan. """

        return next(iter(self.samples.values())).shape[0]

    def __len__(self) -> int:
        """ Number of materialized parameter draws (parameter identities) of this plan. """

        return self.num_draws

    def validate(self, model, dtype: torch.dtype = None) -> None:
        """
        Assert that this plan is still applicable to the given fitted model.

        Parameters
        ----------
        model : AbstractLogisticRegression
            Calibration model on which the plan should be applied.
        dtype : torch.dtype, optional
            Data type of the input data passed to ``transform``. If given, it must match the dtype of the fit
            for which the plan was created.

        Raises
        ------
        RuntimeError
            If the model was refitted/cleared after plan creation, the inference method does not match,
            the parameter structure changed or the data type does not match.
        """

        # plan was created from another model or the model was refitted in the meantime
        if getattr(model, '_model_uid', None) != self.model_uid:
            raise RuntimeError(
                "Sampling plan is invalid: it was created by a different calibration model instance and "
                "cannot be applied to this model. Create a new sampling plan from the current model instead.")

        if getattr(model, '_fit_generation', None) != self.fit_generation:
            raise RuntimeError(
                "Sampling plan is invalid: the model has been (re-)fitted or cleared after the plan was created. "
                "Create a new sampling plan from the current model state instead of reusing a stale plan.")

        if model.method != self.method:
            raise RuntimeError(
                "Sampling plan method '%s' does not match current model method '%s'."
                % (self.method, model.method))

        if model._sites is None or set(model._sites.keys()) != set(self.samples.keys()):
            raise RuntimeError(
                "Sampling plan parameter structure does not match the current model sites. "
                "Expected sites %s but plan contains %s."
                % (list(model._sites.keys()) if model._sites is not None else None,
                   list(self.samples.keys())))

        # every site must still have the same parameter dimension as at plan creation time
        for name, tensor in self.samples.items():
            current_shape = tuple(tensor.shape[1:])
            if tuple(self.site_structure[name]) != current_shape:
                raise RuntimeError(
                    "Internal sampling plan inconsistency for site '%s': recorded structure %s does not match "
                    "stored draw shape %s." % (name, self.site_structure[name], current_shape))

            init_mean = model._sites[name]['init']['mean']

            # compare by number of parameters: MCMC chain samples of a single-element site can be stored
            # as scalars (shape ()) whereas the initialization keeps a length-1 vector - both are
            # interchangeable in the model because a scalar broadcasts to a length-1 vector
            if int(torch.as_tensor(init_mean).numel()) != int(tensor[0].numel()):
                raise RuntimeError(
                    "Sampling plan parameter structure for site '%s' with %d parameter(s) does not match the "
                    "current model parameter shape %s. The model has changed since plan creation."
                    % (name, tensor[0].numel(), tuple(init_mean.shape)))

        # MCMC additionally relies on the stored chain samples from which the indices were drawn
        if self.method == 'mcmc':
            if model.mcmc_model is None:
                raise RuntimeError("Sampling plan requires an MCMC model but the current model has none.")

            for name, tensor in self.samples.items():
                if name not in model.mcmc_model:
                    raise RuntimeError(
                        "MCMC site '%s' recorded in the plan is missing in the current model." % name)

        # a plan created for float32 must not be applied to float64 input and vice versa
        if dtype is not None and dtype != self.dtype:
            raise RuntimeError(
                "Sampling plan was created for dtype %s but is applied to data with dtype %s. "
                "Refit the model or create a matching plan instead of mixing precisions."
                % (self.dtype, dtype))

    def apply(self, model, X):
        """
        Convenience wrapper applying this plan to input data via
        :meth:`~netcal.scaling.AbstractLogisticRegression.transform`.

        Parameters
        ----------
        model : AbstractLogisticRegression
            Fitted calibration model that created the plan.
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values that are calibrated using the fixed parameter identities.

        Returns
        -------
        np.ndarray, shape=(num_draws, n_samples, [n_classes])
            Calibrated estimates with the leading draw dimension in the fixed order defined by this plan.
        """

        return model.transform(X, sampling_plan=self)

    def __getstate__(self) -> Dict:
        """ Support pickling by returning the full instance state. """

        return {
            'method': self.method,
            'samples': self.samples,
            'fit_generation': self.fit_generation,
            'model_uid': self.model_uid,
            'dtype': self.dtype,
            'site_structure': self.site_structure,
            'random_state': self.random_state,
            'mcmc_indices': self.mcmc_indices,
        }

    def __setstate__(self, state: Dict):
        """ Restore instance state after unpickling. """

        self.__init__(
            method=state['method'],
            samples=state['samples'],
            fit_generation=state['fit_generation'],
            model_uid=state['model_uid'],
            dtype=state['dtype'],
            site_structure=state['site_structure'],
            random_state=state['random_state'],
            mcmc_indices=state['mcmc_indices'],
        )
