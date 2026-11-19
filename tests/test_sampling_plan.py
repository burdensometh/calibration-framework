# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
Tests for the reusable posterior sampling plan used by Bayesian (VI/MCMC) logistic calibration.

The plan separates the random draw of posterior parameter identities from the deterministic application
of fixed parameter vectors to input data. These tests verify that:

* chunked application of one plan equals a single full-batch application (up to floating point tolerance),
* independent chunked transforms without a plan do NOT keep parameter identities aligned,
* plans are bound to the fitting generation, method, dtype and parameter structure and reject stale use,
* the plan-based output still feeds the existing uncertainty (mean/variance) error path,
* plans are picklable for reuse in worker processes.
"""

import os
import pickle
import unittest
import warnings

import numpy as np
import torch

from netcal.scaling import (
    LogisticCalibration,
    TemperatureScaling,
    BetaCalibration,
    PosteriorSamplingPlan,
)

torch.set_num_threads(2)
warnings.filterwarnings("ignore")


def _binary_data(n: int = 500, seed: int = 42):
    """ Generate synthetic binary confidence / label data. """

    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.05, 0.95, size=n)
    y = (conf + rng.normal(0.0, 0.15, size=n) > 0.5).astype(int)
    return conf, y


def _multiclass_data(n: int = 600, num_classes: int = 3, seed: int = 7):
    """ Generate synthetic multiclass softmax confidence / label data. """

    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, num_classes))
    conf = _softmax(logits)
    y = rng.integers(0, num_classes, size=n)
    return conf, y


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class TestPosteriorSamplingPlanVI(unittest.TestCase):
    """ Sampling plan behaviour on Variational Inference models. """

    @classmethod
    def setUpClass(cls):
        cls.conf, cls.y = _binary_data()
        cls.model = LogisticCalibration(method='variational', vi_epochs=200)
        cls.model.fit(cls.conf, cls.y, random_state=0, tensorboard=False)

        cls.rng = np.random.default_rng(99)
        cls.X = cls.rng.uniform(0.05, 0.95, size=150)

    def test_plan_attributes(self):
        """ The plan materializes fixed posterior identities with the expected metadata. """

        plan = self.model.create_sampling_plan(num_samples=40, random_state=1)

        self.assertIsInstance(plan, PosteriorSamplingPlan)
        self.assertEqual(len(plan), 40)
        self.assertEqual(plan.num_draws, 40)
        self.assertEqual(plan.method, 'variational')
        self.assertEqual(plan.fit_generation, self.model._fit_generation)
        self.assertEqual(plan.dtype, torch.float64)
        self.assertEqual(set(plan.samples.keys()), set(self.model._sites.keys()))

        # every draw tensor lives on CPU for device-independent reuse
        for tensor in plan.samples.values():
            self.assertEqual(tensor.device.type, 'cpu')
            self.assertEqual(tensor.shape[0], 40)

    def test_chunked_equals_full_batch(self):
        """ Concatenating chunked plan applications matches the full-batch transform within tolerance. """

        plan = self.model.create_sampling_plan(num_samples=60, random_state=3)

        full = self.model.transform(self.X, sampling_plan=plan)

        for n_chunks in (2, 3, 5):
            chunks = np.array_split(self.X, n_chunks)
            stitched = np.concatenate(
                [self.model.transform(chunk, sampling_plan=plan) for chunk in chunks],
                axis=1,
            )

            self.assertEqual(full.shape, stitched.shape)
            np.testing.assert_allclose(stitched, full, rtol=1e-10, atol=1e-8)

    def test_same_draw_identity_across_chunks(self):
        """ Draw index k must be produced by the identical parameter vector on every chunk. """

        plan = self.model.create_sampling_plan(num_samples=25, random_state=11)

        first_half = self.model.transform(self.X[:75], sampling_plan=plan)
        second_half = self.model.transform(self.X[75:], sampling_plan=plan)
        full = self.model.transform(self.X, sampling_plan=plan)

        # parameter identities are identical; only the order of floating point reductions differs,
        # hence results match up to machine precision rather than bit-exact equality
        np.testing.assert_allclose(full[:, :75], first_half, atol=1e-12)
        np.testing.assert_allclose(full[:, 75:], second_half, atol=1e-12)

    def test_plan_is_deterministic_and_seed_independent_at_apply(self):
        """ Applying a plan must not depend on any random state - the identities are fixed. """

        plan = self.model.create_sampling_plan(num_samples=30, random_state=5)
        ref = self.model.transform(self.X, sampling_plan=plan)

        # consume RNG state heavily between applications
        torch.manual_seed(1234)
        np.random.seed(1234)
        _ = torch.randn(1000)
        _ = np.random.randn(1000)

        again = self.model.transform(self.X, sampling_plan=plan, random_state=999)
        np.testing.assert_array_equal(ref, again)

    def test_plan_reproducible_creation(self):
        """ Two plans created with the same seed materialize identical parameter identities. """

        plan_a = self.model.create_sampling_plan(num_samples=35, random_state=21)
        plan_b = self.model.create_sampling_plan(num_samples=35, random_state=21)

        for name in plan_a.samples:
            np.testing.assert_array_equal(
                plan_a.samples[name].numpy(), plan_b.samples[name].numpy())

    def test_independent_chunks_without_plan_misalign(self):
        """ Without a plan, per-chunk draws are different identities, so sample correlation breaks. """

        num_draws = 50
        full = self.model.transform(self.X, num_samples=num_draws, random_state=0)

        # each chunk performs its own independent transform without a shared plan.
        # note that no random_state is passed - merely reusing the same seed on every chunk would hide
        # the correlation problem by RNG coincidence, which the sampling plan explicitly avoids relying on.
        stitched = np.concatenate([
            self.model.transform(chunk, num_samples=num_draws)
            for chunk in np.array_split(self.X, 3)
        ], axis=1)

        self.assertEqual(full.shape, stitched.shape)
        # correlations are not preserved: at least some draws must differ between the two schemes
        self.assertGreater(np.max(np.abs(full - stitched)), 1e-3)

    def test_uncertainty_error_path_consistency(self):
        """ Plan output keeps the (draw, data, ...) layout and yields consistent mean/variance statistics. """

        plan = self.model.create_sampling_plan(num_samples=80, random_state=4)
        full = self.model.transform(self.X, sampling_plan=plan)
        stitched = np.concatenate([
            self.model.transform(chunk, sampling_plan=plan)
            for chunk in np.array_split(self.X, 4)
        ], axis=1)

        np.testing.assert_allclose(full.mean(axis=0), stitched.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(full.var(axis=0), stitched.var(axis=0), atol=1e-10)

    def test_plan_rejected_after_refit(self):
        """ A plan created before a refit is rejected by generation mismatch rather than silently reused. """

        plan = self.model.create_sampling_plan(num_samples=20, random_state=2)
        generation_before = self.model._fit_generation

        self.model.fit(self.conf, self.y, random_state=1, tensorboard=False)
        self.assertGreater(self.model._fit_generation, generation_before)

        with self.assertRaises(RuntimeError):
            self.model.transform(self.X, sampling_plan=plan)

        with self.assertRaises(RuntimeError):
            plan.validate(self.model)

        # a fresh plan on the refitted model works again
        new_plan = self.model.create_sampling_plan(num_samples=20, random_state=2)
        out = self.model.transform(self.X, sampling_plan=new_plan)
        self.assertEqual(out.shape[0], 20)

    def test_plan_rejected_after_clear(self):
        """ Clearing the model invalidates the plan as well. """

        model = LogisticCalibration(method='variational', vi_epochs=50)
        model.fit(self.conf, self.y, random_state=0, tensorboard=False)
        plan = model.create_sampling_plan(num_samples=10)

        model.clear()
        with self.assertRaises(RuntimeError):
            model.transform(self.X, sampling_plan=plan)

    def test_plan_cannot_apply_to_other_model(self):
        """ A plan from one model must not be accepted by a differently fitted model. """

        plan = self.model.create_sampling_plan(num_samples=15, random_state=6)
        other = LogisticCalibration(method='variational', vi_epochs=50)
        other.fit(self.conf, self.y, random_state=123, tensorboard=False)

        with self.assertRaises(RuntimeError):
            other.transform(self.X, sampling_plan=plan)

    def test_plan_incompatible_with_mean_estimate(self):
        """ Combining a plan with mean_estimate must raise instead of dropping the identities silently. """

        plan = self.model.create_sampling_plan(num_samples=10)
        with self.assertRaises(ValueError):
            self.model.transform(self.X, mean_estimate=True, sampling_plan=plan)

    def test_pickle_roundtrip(self):
        """ Plans are picklable and remain valid after unpickling for use in worker processes. """

        plan = self.model.create_sampling_plan(num_samples=25, random_state=8)
        restored = pickle.loads(pickle.dumps(plan))

        self.assertEqual(restored.fit_generation, plan.fit_generation)
        self.assertEqual(restored.num_draws, plan.num_draws)

        ref = self.model.transform(self.X, sampling_plan=plan)
        out = self.model.transform(self.X, sampling_plan=restored)
        np.testing.assert_array_equal(ref, out)

    def test_plan_apply_convenience(self):
        """ The plan's apply wrapper delegates to transform with the fixed identities. """

        plan = self.model.create_sampling_plan(num_samples=20, random_state=14)
        direct = self.model.transform(self.X, sampling_plan=plan)
        via_apply = plan.apply(self.model, self.X)
        np.testing.assert_array_equal(direct, via_apply)

    def test_plan_rejected_on_mle(self):
        """ MLE models neither create nor accept sampling plans. """

        mle = LogisticCalibration(method='mle')
        mle.fit(self.conf, self.y)
        with self.assertRaises(RuntimeError):
            mle.create_sampling_plan(num_samples=10)

        plan = self.model.create_sampling_plan(num_samples=10)
        with self.assertRaises(RuntimeError):
            mle.transform(self.X, sampling_plan=plan)


class TestPosteriorSamplingPlanMulticlass(unittest.TestCase):
    """ Sampling plan on multiclass models, including 3-D prepared data (Beta calibration). """

    @classmethod
    def setUpClass(cls):
        cls.conf, cls.y = _multiclass_data()
        cls.rng = np.random.default_rng(321)
        cls.X = _softmax(cls.rng.normal(size=(120, 3)))

    def test_vector_scaling_chunk_consistency(self):
        model = LogisticCalibration(method='variational', vi_epochs=120)
        model.fit(self.conf, self.y, random_state=0, tensorboard=False)

        plan = model.create_sampling_plan(num_samples=30, random_state=2)
        full = model.transform(self.X, sampling_plan=plan)
        stitched = np.concatenate([
            model.transform(chunk, sampling_plan=plan)
            for chunk in np.array_split(self.X, 4)
        ], axis=1)

        self.assertEqual(full.shape, (30, 120, 3))
        np.testing.assert_allclose(stitched, full, rtol=1e-10, atol=1e-8)

    def test_beta_calibration_chunk_consistency(self):
        model = BetaCalibration(method='variational', vi_epochs=120)
        model.fit(self.conf, self.y, random_state=0, tensorboard=False)

        plan = model.create_sampling_plan(num_samples=25, random_state=5)
        full = model.transform(self.X, sampling_plan=plan)
        stitched = np.concatenate([
            model.transform(chunk, sampling_plan=plan)
            for chunk in np.array_split(self.X, 3)
        ], axis=1)

        self.assertEqual(full.shape, (25, 120, 3))
        np.testing.assert_allclose(stitched, full, rtol=1e-10, atol=1e-8)


class TestPosteriorSamplingPlanTemperature(unittest.TestCase):
    """ Temperature scaling has a single weights site and no bias site. """

    def test_temperature_scaling_plan(self):
        conf, y = _binary_data()
        model = TemperatureScaling(method='variational', vi_epochs=120)
        model.fit(conf, y, random_state=0, tensorboard=False)

        X = np.random.default_rng(5).uniform(0.05, 0.95, size=80)
        plan = model.create_sampling_plan(num_samples=20, random_state=9)

        self.assertEqual(list(plan.samples.keys()), ['weights'])

        full = model.transform(X, sampling_plan=plan)
        stitched = np.concatenate([
            model.transform(chunk, sampling_plan=plan)
            for chunk in np.array_split(X, 2)
        ], axis=1)

        np.testing.assert_allclose(stitched, full, rtol=1e-10, atol=1e-8)


class TestPosteriorSamplingPlanMCMC(unittest.TestCase):
    """ Sampling plan behaviour on MCMC models (small chain for runtime). """

    @classmethod
    def setUpClass(cls):
        cls.conf, cls.y = _binary_data(n=200)
        cls.model = LogisticCalibration(
            method='mcmc', mcmc_steps=40, mcmc_warmup_steps=20, mcmc_chains=1)
        cls.model.fit(cls.conf, cls.y, random_state=0, tensorboard=False)
        cls.X = np.random.default_rng(77).uniform(0.05, 0.95, size=90)

    def test_mcmc_plan_stores_indices_and_identities(self):
        plan = self.model.create_sampling_plan(num_samples=30, random_state=3)

        self.assertEqual(plan.method, 'mcmc')
        self.assertIsNotNone(plan.mcmc_indices)
        self.assertEqual(plan.mcmc_indices.shape, (30,))
        self.assertTrue(torch.all(plan.mcmc_indices < self.model.mcmc_steps))

    def test_mcmc_chunk_consistency(self):
        plan = self.model.create_sampling_plan(num_samples=25, random_state=4)
        full = self.model.transform(self.X, sampling_plan=plan)
        stitched = np.concatenate([
            self.model.transform(chunk, sampling_plan=plan)
            for chunk in np.array_split(self.X, 3)
        ], axis=1)

        self.assertEqual(full.shape, stitched.shape)
        np.testing.assert_allclose(stitched, full, rtol=1e-10, atol=1e-8)

    def test_mcmc_plan_matches_manual_index_selection(self):
        """ The plan's materialized draws must be exactly the indexed MCMC chain samples. """

        plan = self.model.create_sampling_plan(num_samples=20, random_state=6)
        for name, chain in self.model.mcmc_model.items():
            expected = chain.index_select(0, plan.mcmc_indices.to(chain.device))
            np.testing.assert_array_equal(
                plan.samples[name].numpy(),
                torch.squeeze(expected, dim=1).cpu().numpy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
