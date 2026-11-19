# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

import abc, os
import numpy as np
from typing import Tuple, List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn.functional as F

from netcal import __version__ as version
from .Helper import squeeze_generic
from .Decorator import accepts, dimensions
from .LabelObservation import LabelObservationError, prepare_label_matrix, class_status


try:
    import cPickle as pickle
except ImportError:
    import pickle


class AbstractCalibration(BaseEstimator, TransformerMixin):
    """
    Abstract base class for all calibration methods.
    Inherits functions from sklearn's BaseEstimator.

    Parameters
    ----------
    detection : bool, default: False
        If False, the input array 'X' is treated as multi-class confidence input (softmax)
        with shape (n_samples, [n_classes]).
        If True, the input array 'X' is treated as a box predictions with several box features (at least
        box confidence must be present) with shape (n_samples, [n_box_features]).
    independent_probabilities : bool, optional, default: False
        Boolean for multi class probabilities.
        If set to True, the probability estimates for each
        class are treated as independent of each other (sigmoid).

    Attributes
    ----------
    epsilon : float
        Lowest possible digit that can be computed. Needed for several operations like divisions or log to guarantee
        values inequal to 0 or 1.
    """

    @accepts(bool, bool)
    def __init__(self, detection: bool = False, independent_probabilities: bool = False):
        """
        Create an instance of `AbstractCalibration`.

        Parameters
        ----------
        detection : bool, default: False
            If False, the input array 'X' is treated as multi-class confidence input (softmax)
            with shape (n_samples, [n_classes]).
            If True, the input array 'X' is treated as a box predictions with several box features (at least
            box confidence must be present) with shape (n_samples, [n_box_features]).
        independent_probabilities : bool, optional, default: False
            Boolean for multi class probabilities.
            If set to True, the probability estimates for each
            class are treated as independent of each other (sigmoid).
        """

        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)

        self.detection = detection
        self.num_classes = None
        self.independent_probabilities = independent_probabilities

        # this one is for 'clear' method to restore default
        self._default_independent_probabilities = independent_probabilities

        # label observation mask contract - populated by 'fit' in independent
        # multi-label mode (or on binary/detection data with unknown labels)
        self.label_support_ = None
        self.label_status_ = {}
        self.class_status_ = None
        self.fitted_classes_ = []
        self.unfittable_classes_ = []
        self.unobserved_classes_ = []
        self.degenerate_classes_ = []
        self.fallback_classes_ = []
        self.support_ = None

        # canonical label observation (labels + mask) used to decide every per-class
        # denominator. This is metadata and never part of the feature matrix.
        self._label_observation = None
        self._multilabel = False

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract function call to build the calibration model.
        This function performs several checks and returns the improved X and y.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.
        y : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with ground truth labels.
            Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
            In independent multi-label mode (``independent_probabilities=True``), a 2-D array
            with binary entries per class is expected. Unknown labels must be encoded as
            ``numpy.nan`` or supplied through the ``mask`` parameter - they are *not* negatives.
            ``numpy.ma.MaskedArray`` inputs are supported as well.
        mask : np.ndarray of bool, shape=(n_samples, [n_classes]), optional, default: None
            Label observation mask. ``True`` marks a label as known, ``False`` as unknown.
            A 1-D mask is valid for binary classification (one entry per sample). The mask is
            metadata only and is never appended to or otherwise mixed into the feature matrix
            ``X``.

        Returns
        -------
        tuple
            Checked and converted X and y. In independent multi-label mode, ``y`` is a
            2-D array with ``numpy.nan`` for unknown labels.

        Raises
        ------
        AttributeError
            - If 'X' has more than 2 dimensions
            - If 'y' has more than 2 dimensions
            - If 'y' as one-hot encoded ground truth does not match to detected number of classes
        LabelObservationError
            If the label observation mask violates the contract (for instance, if an entry is
            marked as observed but contains no valid binary label).
        """

        # invoke clear method on each fit
        self.clear()

        # check number of given samples
        if X.size <= 0:
            raise ValueError("No X samples provided.")

        # check number of given samples
        if y.size <= 0:
            raise ValueError("No y samples provided.")

        if y.shape[0] != X.shape[0]:
            raise AttributeError('Number of samples given by \'X\' and \'y\' is not equal.')

        # -----------------------------------------------------------------
        # preprocessing of confidence values given with X

        # remove single-dimensional entries if present
        X = squeeze_generic(X, axes_to_keep=0)

        # check shape of input array X and determine number of classes
        # first case: confidence array is 1-D: binary classification problem
        # this is either detection or classification mode (not important)
        if len(X.shape) == 1:
            self.num_classes = 2
            self.independent_probabilities = False

            # on detection mode, two dimensions are mandatory
            if self.detection:
                X = np.reshape(X, (-1, 1))

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

            # on detection, we face a binary classification problem
            if self.detection:
                self.num_classes = 2

            # classification
            else:

                # number of classes is length of second dimension (softmax) but at least 2 classes (binary case)
                self.num_classes = max(2, X.shape[1])

                # if second dimensions is less or equal than 2 and inputs are independent (multiple sigmoid outputs)
                # treat as binary classification and extract confidence estimates for y=1
                if X.shape[1] <= 2 and not self.independent_probabilities:
                    X = X[:, -1]
        else:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'X\' with num dimensions of %d."
                                 % len(X.shape))

        # check if array's values are in interval [0, 1]. This is not allowed for classification
        # and detection as well
        if not ((X >= 0).all() and (X <= 1).all()):
            raise ValueError("Some values of \'X\' are not in range [0, 1].")

        # -----------------------------------------------------------------
        # preprocessing of ground truth values given with y

        # remove single-dimensional entries if present
        y = squeeze_generic(y, axes_to_keep=0)

        # Independent (multi-label) classification always uses per-class binary targets, even if
        # every label happens to be known. An explicit observation mask or unknown (NaN/masked)
        # labels route the remaining modes through the same validator.
        if self.independent_probabilities and not self.detection:
            return self._prepare_labelled_data(X, y, mask)

        has_unknown = self._labels_have_unknown(y)

        # single-label multi-class targets can only drop whole rows (unknown is a row property)
        if not self.detection and self.num_classes > 2 and (mask is not None or has_unknown):
            return self._prepare_single_label_data(X, y, mask)

        # binary classification or detection: keep a single binary target
        if self.detection or self.num_classes == 2:
            if mask is not None or has_unknown:
                return self._prepare_labelled_data(X, y, mask)

        # check shape of ground truth array y
        # array is 2-D: assume one-hot encoded
        if len(y.shape) == 2:

            # only binary problem but 2-dimensional and probabilities independent?
            # get labels only for y=1
            if y.shape[1] <= 2 and self.num_classes == 2 and not self.independent_probabilities:
                y = y[:, -1]

            elif y.shape[1] != self.num_classes:
                raise AttributeError("Assuming \'y\' as 2-dimensional one-hot encoded does not match to number of "
                                     "classes found in confidence array \'X\'.")

            else:
                # convert y from one-hot encoded to class labels (1-D)
                y = np.argmax(y, axis=1)

        elif len(y.shape) > 2:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'y\' with num dimensions of %d."
                                 % len(y.shape))

        # count all class labels and warn if not all labels are present
        unique = np.unique(y)

        # on detection, only binary labels {0, 1} are allowed
        # the method "issubset" also allows the presence of only {0} or {1} in the input data
        if self.detection and not set(np.round(unique).astype(np.int64)).issubset({0, 1}):
            raise RuntimeError("On detection mode, it is mandatory to provide binary labels y in {0, 1}.")

        if len(unique) != self.num_classes:
            print("WARNING: Not all class labels are present in ground truth array \'y\'. "
                  "This could led to errors in some models.")

        return X, y

    @staticmethod
    def _labels_have_unknown(y: np.ndarray) -> bool:
        """Return True if the labels carry unknown entries (NaN or a MaskedArray mask)."""

        if isinstance(y, np.ma.MaskedArray):
            return bool(np.ma.getmaskarray(y).any())

        y = np.asarray(y)
        return bool(np.issubdtype(y.dtype, np.floating) and np.isnan(y).any())

    def _prepare_labelled_data(
            self,
            X: np.ndarray,
            y: np.ndarray,
            mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate labels with unknown entries and build the observation-mask state.

        For independent multi-label classification (``independent_probabilities=True``), the
        labels are kept as 2-D array with per-class binary targets and per-class observation
        masks. Every class is fitted from its own observed labels only - rows are *not*
        dropped globally, so known labels of other classes are preserved.

        For binary classification and detection, unknown samples are dropped from the training
        data (there is no other label information within a sample that could be lost).
        """

        # independent multi-label path: one binary model per class
        if not self.detection and self.independent_probabilities:

            # single-label multi-class targets may still be requested with independent
            # probabilities. Convert them to one-vs-all per-class targets *before* validation,
            # so that every class is fully observed and no label is silently turned into a
            # negative by a missing entry. A MaskedArray is preserved because its mask carries
            # the unknown labels.
            if not isinstance(y, np.ma.MaskedArray):
                y = np.asarray(y)

                if y.ndim == 1 and self.num_classes > 2:
                    valid = np.isfinite(y)
                    if not np.all(valid):
                        raise LabelObservationError(
                            "1-D multi-class labels with unknown entries cannot be expanded to per-class "
                            "targets. Pass a 2-D label matrix (or a 2-D observation mask) instead."
                        )
                    y = np.eye(self.num_classes)[y.astype(np.int64)]

            observation = prepare_label_matrix(y, mask, num_classes=self.num_classes)
            self._set_observation_state(observation)
            self._multilabel = True

            if not self.fitted_classes_:
                print("WARNING: No class has both binary outcomes observed. "
                      "All classes will fall back to the uncalibrated confidence.")

            return X, observation.labels

        if np.asarray(y).ndim > 2:
            raise AttributeError("Unknown array dimension for parameter \'y\' with num dimensions of %d."
                                 % np.asarray(y).ndim)

        # a 1-D label vector denotes the positive class; the observation state is shared by
        # both classes. Expand it to obtain a class-wise support/status record.
        observation = prepare_label_matrix(y, mask, num_classes=2)
        self._set_observation_state(observation)
        self._multilabel = False

        # reduce to the positive-class target and drop only the samples without a known label
        observed = observation.mask[:, -1]
        y_binary = observation.labels[observed, -1]

        if self.detection:
            # detection mode requires binary labels {0, 1}
            if not np.all(np.isin(y_binary, (0.0, 1.0))):
                raise RuntimeError("On detection mode, it is mandatory to provide binary labels y in {0, 1}.")

        # Do not postpone filtering: the downstream binary methods expect a 1-D target.
        X = X[observed]
        return X, y_binary

    def _prepare_single_label_data(self, X, y, mask=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate a single-label multi-class target and drop only fully unobserved rows.

        A single-label sample carries exactly one class label, so an unknown label is a
        whole-row property. Per-sample masks and fully-NaN rows are accepted; a 2-D mask that
        marks only some classes of a sample as observed is contradictory and rejected.

        One-hot encoded input without unknown entries keeps the established multi-class path.
        """

        y = np.asarray(y.filled(np.nan) if isinstance(y, np.ma.MaskedArray) else y)

        if y.ndim > 2:
            raise AttributeError("Unknown array dimension for parameter \'y\' with num dimensions of %d."
                                 % y.ndim)

        # 1-D class labels are already in the single-label form
        if y.ndim == 1:
            observed = np.isfinite(y)
            if mask is not None:
                # a single-label sample has exactly one (row-wise) observation state
                observed = self._rowwise_observation(mask, 1, y.shape[0])
            return X[observed], y[observed].astype(np.int64)

        # 2-D one-hot encoded targets
        if mask is not None:
            mask = np.asarray(mask)
            if mask.ndim == 2:
                if mask.shape != y.shape:
                    raise LabelObservationError(
                        "Observation mask has shape %s but labels have shape %s."
                        % (str(mask.shape), str(y.shape))
                    )
                # partially observed per-class mask is contradictory for a single-label target
                rows = mask
                if not np.all(rows.any(axis=1) == rows.all(axis=1)):
                    raise LabelObservationError(
                        "A 2-D observation mask is contradictory for single-label multi-class targets: "
                        "a sample either has its one label observed or not. Use \'independent_probabilities=True\' "
                        "for per-class observations."
                    )
                observed = rows.all(axis=1)
            else:
                observed = self._rowwise_observation(mask, 1, y.shape[0])
        else:
            observed = np.isfinite(y).all(axis=1)

        # unobserved rows are dropped; the known rows keep their one-hot identity
        y_observed = y[observed]
        y_labels = np.argmax(y_observed, axis=1)

        return X[observed], y_labels

    @staticmethod
    def _rowwise_observation(mask, mask_width: int, num_samples: int) -> np.ndarray:
        """Validate a per-sample observation mask and return its boolean vector."""

        mask = np.asarray(mask).astype(bool)

        if mask.ndim == 2:
            if mask.shape[1] != mask_width:
                raise LabelObservationError(
                    "Observation mask has %d columns but %d were expected."
                    % (mask.shape[1], mask_width)
                )
            mask = mask.all(axis=1)

        if mask.shape[0] != num_samples:
            raise LabelObservationError(
                "Observation mask has %d entries but %d samples were provided."
                % (mask.shape[0], num_samples)
            )

        return mask.reshape(-1)

    def _set_observation_state(self, observation):
        """Store the canonical label observation state and expose it as public attributes."""

        self._label_observation = observation
        support = observation.support

        statuses, support = class_status(observation.labels, observation.mask)
        self.support_ = support
        self.class_status_ = statuses

        self.fitted_classes_ = [k for k, status in enumerate(statuses) if status == 'fitted']
        self.unfittable_classes_ = [k for k, status in enumerate(statuses) if status != 'fitted']

        # aliases that make the status explicit for users without knowing the private names
        self.label_support_ = support
        self.label_status_ = {k: status for k, status in enumerate(statuses)}
        self.unobserved_classes_ = [k for k, status in enumerate(statuses) if status == 'unobserved']
        self.degenerate_classes_ = [k for k, status in enumerate(statuses) if status == 'degenerate']
        self.fallback_classes_ = list(self.unfittable_classes_)

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Abstract function. After model calibration, this function is used to get calibrated outputs of uncalibrated
        confidence estimates.
        This function performs several checks and returns the improved X.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, [n_classes]) or (n_samples, [n_box_features])
            NumPy array with confidence values for each prediction on classification with shapes
            1-D for binary classification, 2-D for multi class (softmax).
            On detection, this array must have 2 dimensions with number of additional box features in last dim.

        Returns
        -------
        np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with uncalibrated but checked confidence estimates.
            1-D for binary classification, 2-D for multi class (softmax).

        Raises
        ------
        RuntimeError
            If method is called before 'fit'.
        AttributeError
            - If model was built for multi class but 2nd dimension of 'X' does not match to number of classes.
            - If 'X' has more than 2 dimensions
        """

        # transform method invoked but model not built? Raise error
        if self.num_classes is None:
            raise RuntimeError('Could not invoke \'transform\' method before fit.')

        # -----------------------------------------------------------------
        # preprocessing of confidence values given with X

        # remove single-dimensional entries if present
        X = squeeze_generic(X, axes_to_keep=0)

        # got only 1-D array but model was fit for more than 2 classes?
        if len(X.shape) == 1:

            if self.num_classes > 2 and not self.detection:
                raise AttributeError("Model was build for %d classes but 1-D confidence array was provided."
                                     % self.num_classes)

        # second case: confidence array is 2-D: binary or multi class classification problem
        elif len(X.shape) == 2:

            if X.shape[1] != self.num_classes and not self.detection:
                raise AttributeError("Model was build for %d classes but 2-D confidence  array  with %d classes"
                                     " was provided."
                                     % (self.num_classes, X.shape[1]))

            # if second dimensions is less or equal than 2 and inputs are independent (multiple sigmoid outputs)
            # treat as binary classification and extract confidence estimates for y=1
            if X.shape[1] == self.num_classes == 2 and not self.independent_probabilities and not self.detection:
                X = X[:, -1]

        else:

            # unknown shape
            raise AttributeError("Unknown array dimension for parameter \'X\' with num dimensions of %d."
                                 % len(X.shape))

        return X

    @abc.abstractmethod
    def clear(self):
        """
        Clear model parameters.
        """

        self.num_classes = 2 if self.detection else None
        self.independent_probabilities = self._default_independent_probabilities

        # reset the label observation state
        self.label_support_ = None
        self.label_status_ = {}
        self.class_status_ = None
        self.fitted_classes_ = []
        self.unfittable_classes_ = []
        self.unobserved_classes_ = []
        self.degenerate_classes_ = []
        self.fallback_classes_ = []
        self.support_ = None

        self._label_observation = None
        self._multilabel = False

    def epsilon(self, dtype: Union[np.dtype, torch.dtype]) -> float:
        """
        Get the smallest digit that is representable depending on the passed dtype (NumPy or PyTorch).

        Parameters
        ----------
        dtype : either NumPy dtype or PyTorch dtype
            Data type.

        Returns
        -------
        float
            Smallest representable digit depending on the passed input dtype.
        """

        if isinstance(dtype, torch.dtype):
            return float(torch.finfo(dtype).eps)
        elif isinstance(dtype, np.dtype):
            return float(np.finfo(dtype).eps)
        else:
            raise RuntimeError("Unknown dtype:", dtype)

    def get_params(self, deep=True) -> Dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        # get all params of current instance and save as dict
        params = dict(vars(self))

        if deep:

            # needed for all binary methods that are distributed to multi class by one-vs-all
            if hasattr(self, '_multiclass_instances'):
                params['_multiclass_instances'] = []

                for i, instance in self._multiclass_instances:
                    params['_multiclass_instances'].append((i, instance.get_params(deep=True)))

        return params

    def set_params(self, **params) -> 'AbstractCalibration':
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """

        if '_multiclass_instances' in params:
            self._multiclass_instances = []

            for label, subparams in params['_multiclass_instances']:
                instance = self.__class__()
                instance.set_params(**subparams)

                self._multiclass_instances.append((label, instance))

            # remove key and value from dict to prevent override in super method
            del params['_multiclass_instances']

        for key, value in params.items():
            setattr(self, key, value)

        return self

    @accepts(str)
    def save_model(self, filename: str):
        """
        Save model instance as with torch's save function as this is safer for torch tensors.

        Parameters
        ----------
        filename : str
            String with filename.
        """

        if filename.rfind("/") > 0:
            dir_path = filename[:filename.rfind("/")]
            os.makedirs(dir_path, exist_ok=True)

        with open(filename, 'wb') as write_object:
            torch.save(self.get_params(deep=True), write_object, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        """
        Load model from saved torch dump.

        Parameters
        ----------
        filename : str
            String with filename.

        Returns
        -------
        AbstractCalibration
            Instance of a child class of `AbstractCalibration`.
        """

        try:
            with open(filename, 'rb') as read_object:
                params = torch.load(read_object)
        except RuntimeError:
            raise IOError("Stored models of version <1.1 are not compatible with current version %s" % version)

        self.set_params(**params)
        return self

    @accepts(np.ndarray, np.ndarray, list, str)
    def _calc_model_scores(self, confidences: np.ndarray, ground_truth: np.ndarray,
                           model_list: List['AbstractCalibration'], score_function: str = 'BIC') -> np.ndarray:
        """
        Calculates the Bayesian Scores for each Histogram Binning model and discards each model which
        gets a score of 0 to speed up predictions later on.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        model_list : list
            List with models to compute the scores for.
        score_function : str, default: 'BIC'
            Score function which to use. Must be either 'BIC' or 'AIC'.

        Returns
        -------
        np.ndarray, shape=(n_models,)
            NumPy 1-D array with scores for each model provided in list.
        """

        num_samples = ground_truth.size

        # calculate log likelihood of confidences
        log_likelihood = np.zeros(len(model_list))
        for i, model in enumerate(model_list):
            estimate = model.transform(confidences)

            # use torch routines to get log likelihood
            if self._is_binary_classification():
                loss = F.binary_cross_entropy(torch.from_numpy(estimate).type(torch.FloatTensor),
                                              torch.from_numpy(ground_truth).type(torch.FloatTensor),
                                              reduction='sum')

            # on multiclass, use NLLLoss - this function expects log-probabilities
            else:
                loss = F.nll_loss(torch.from_numpy(np.log(estimate)).type(torch.FloatTensor),
                                  torch.from_numpy(ground_truth).type(torch.FloatTensor),
                                  reduction='sum')

            log_likelihood[i] = -loss.item()

        # get degrees of freedom of each model. This is equivalent to number of groups
        degrees_of_freedom = np.array([x.get_degrees_of_freedom() for x in model_list], dtype=np.int32)

        # choose scoring function 'AIC' oder 'BIC'
        if score_function == 'aic':
            score = 2. * degrees_of_freedom - 2. * log_likelihood
        elif score_function == 'bic':
            score = -2. * log_likelihood + degrees_of_freedom * np.log(num_samples)
        else:
            raise ValueError("Unknown score function \'%s\'. Fix your implementation")

        # calculate relative likelihood of each model and normalize scores
        model_scores = np.exp((np.min(score) - score) / 2.)

        return model_scores

    @accepts(np.ndarray, np.ndarray, list, str, float)
    def _elbow(self, confidences: np.ndarray, ground_truth: np.ndarray, model_list: List['AbstractCalibration'],
               score_function: str = 'BIC', alpha: float = 0.001) -> Tuple[List[float], List['AbstractCalibration']]:
        """
        Select models by Bayesian score and discard models below a certain threshold with elbow method.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, [n_classes])
            NumPy array with confidence values for each prediction.
            1-D for binary classification, 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with ground truth labels.
        model_list : list
            List with models to compute the scores for.
        score_function : str, default: 'BIC'
            Score function which to use. Must be either 'BIC' or 'AIC'.
        alpha : float, default: 0.001
            Threshold of model scores.

        Returns
        -------
        tuple, shape=(2,)
            Tuple with two lists of (normed) model scores and list of models.
        """

        model_scores = self._calc_model_scores(confidences, ground_truth, model_list, score_function)

        num_models = len(model_scores)
        score_variance = np.var(model_scores)

        # get reversed sorted indices
        sorted_indices = np.argsort(model_scores)[::-1]
        sorted_model_scores = model_scores[sorted_indices]

        # k is number of models kept
        k = 0

        # first part: seek for firt models with equal score
        while k < (num_models-1) and sorted_model_scores[k] == sorted_model_scores[k+1]:
            k += 1

        # second part: seek for models with score difference of two consecutive models scaled by
        # score variance below threshold
        while k < (num_models-1) and ((sorted_model_scores[k] - sorted_model_scores[k+1]) / score_variance) > alpha:
            k += 1

        # k denotes amount of kept models - range(k) must also include last model
        # thus, increase by 1
        k += 1

        kept_models = [model_list[sorted_indices[i]] for i in range(k)]

        # norm scores to sum up to 1
        kept_scores = [model_scores[sorted_indices[i]] for i in range(k)]
        kept_scores /= np.sum(kept_scores)

        return kept_scores, kept_models

    def _create_one_vs_all_models(self, confidences: np.ndarray, ground_truth: np.ndarray,
                                  model_class: 'AbstractCalibration', *constructor_args,
                                  model_factory=None) -> List['AbstractCalibration']:
        """
        Create for K classes K one vs all calibration models.

        In independent multi-label mode, every class is fitted from its *own observed*
        labels only. Rows are never dropped globally, so known labels of other classes are
        preserved. Classes that are unobserved or degenerate are not fitted at all; they are
        recorded in :attr:`unfittable_classes_` and receive an explicit fallback in
        ``transform`` (the uncalibrated confidence of that class).

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, n_classes)
            NumPy array with confidence values for each prediction.
            This must be 2-D for multi class (softmax).
        ground_truth : np.ndarray, shape=(n_samples,) or (n_samples, n_classes)
            NumPy 1-D array with ground truth labels (one-vs-all is derived per class) or a
            2-D array with per-class binary labels in independent multi-label mode. Unknown
            labels in the 2-D case are encoded as ``numpy.nan``.
        model_class : child of AbstractCalibration
            instance of child of `AbstractCalibration` class.
        *constructor_args
            several args passed to class constructor.
        model_factory : callable, optional, default: None
            Callable that returns a fresh *binary* calibration model. Defaults to
            ``model_class(*constructor_args, independent_probabilities=...)``. Used by the
            regression-based scaling methods to spawn child instances with identical params.

        Returns
        -------
        list, shape=(n_classes,)
            List with calibration models.
        """

        multiclass_instances = []

        # independent multi-label data: per-class binary labels with a per-class observation mask
        if ground_truth.ndim == 2:
            observation = self._label_observation
            if observation is None:
                raise LabelObservationError("2-D labels require a label observation state built by \'fit\'.")

            for label in range(self.num_classes):
                status = self.class_status_[label]

                if status != 'fitted':
                    print("WARNING: class %d is \'%s\'; no calibration model is fitted and the uncalibrated "
                          "confidence is used as explicit fallback." % (label, status))
                    continue

                # only use samples whose label for *this* class has been observed
                observed = observation.mask[:, label]
                onevsall_confidence = self._get_one_vs_all_confidence(confidences, label)[observed]
                onevsall_ground_truth = observation.labels[observed, label]

                # build an own binary calibration model for each label; the mask is metadata and
                # is never appended to the confidence features
                if model_factory is not None:
                    model = model_factory()
                    # the regression-based scaling methods log to tensorboard by default; per-class
                    # child fits must not spam separate run directories
                    model.fit(onevsall_confidence, onevsall_ground_truth, tensorboard=False)
                else:
                    model = model_class(*constructor_args, independent_probabilities=False)
                    model.fit(onevsall_confidence, onevsall_ground_truth)
                multiclass_instances.append(tuple((label, model)))

            return multiclass_instances

        for label in range(self.num_classes):

            if np.where(ground_truth == label)[0].size == 0:
                print("WARNING: no training data for label %d present" % label)
                continue

            # get 1 vs all vector depending on current label
            onevsall_ground_truth = self._get_one_vs_all_label(ground_truth, label)
            onevsall_confidence = self._get_one_vs_all_confidence(confidences, label)

            # build an own Histogram Binning model for each label in a 1 vs all manner
            # now it's a k-fold binary classification task
            model = model_class(*constructor_args, independent_probabilities=self.independent_probabilities)
            model.fit(onevsall_confidence, onevsall_ground_truth)

            # add instances to internal list for calibrating new confidence estimates
            multiclass_instances.append(tuple((label, model)))

        return multiclass_instances

    def _transform_one_vs_all(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the per-class calibration models and keep a full class axis in the result.

        Classes without a fitted model - because they are unobserved, insufficient or
        degenerate - fall back explicitly to their uncalibrated confidence. The mask is never
        part of the returned features; every class keeps its own identity (column).

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_classes)
            Confidence estimates to calibrate.

        Returns
        -------
        np.ndarray, shape=(n_samples, n_classes)
            Calibrated confidences on the full class axis.
        """

        # start from zeros; classes without a fitted model keep their explicit fallback below
        calibrated = np.zeros_like(X, dtype=np.float64)

        for label, model in self._multiclass_instances:
            onevsall_confidence = self._get_one_vs_all_confidence(X, label)

            # probabilistic scaling models can return one estimate per sampled parameter set.
            # The class axis must stay one-dimensional, so request the mean estimate.
            if getattr(model, 'method', None) in ('variational', 'mcmc'):
                onevsall_calibrated = model.transform(onevsall_confidence, mean_estimate=True)
            else:
                onevsall_calibrated = model.transform(onevsall_confidence)

            calibrated[:, label] = np.asarray(onevsall_calibrated).reshape(-1)

        # independent multi-label mode: classes without a fitted model fall back explicitly to
        # their uncalibrated confidence. The class axis is kept complete and the mask is never
        # part of the returned features.
        if self._multilabel:
            for label in self.fallback_classes_:
                calibrated[:, label] = X[:, label]

        # normalized multi-class (softmax) output requires a probability sum of 1
        # independent (multi-label) probabilities must *not* be normalized
        elif not self.independent_probabilities:
            normalizer = np.sum(calibrated, axis=1, keepdims=True)
            calibrated = np.divide(
                calibrated, normalizer,
                out=np.zeros_like(calibrated),
                where=normalizer != 0
            )

        return calibrated

    @classmethod
    def _sort_arrays(cls, array1: np.ndarray, *args) -> Tuple:
        """
        Sort multiple NumPy arrays by values given with array1.

        Parameters
        ----------
        array1 : np.ndarray
            NumPy 1-D array with values to sort.
        *args : np.ndarray
            NumPy 1-D arrays that get sorted by array1.

        Returns
        -------
        tuple, shape=(n_args,)
            Tuple with all given arrays sorted.

        Raises
        ------
        AttributeError
            If types of parameters are not np.ndarray, if shapes of arrays is not 1-D or
            if shapes of arrays do not match to each other.
        """

        if type(array1) != np.ndarray:
            raise AttributeError("Type of arrays must be numpy.ndarray")

        if len(array1.shape) != 1:
            raise AttributeError("Shape of arrays must be 1-D")

        # check dimensions
        for array in args:
            if type(array) != np.ndarray:
                raise AttributeError("Type of arrays must be numpy.ndarray")

            if array1.shape != array.shape:
                raise AttributeError("Shape mismatch of arrays")

        # sort arrays by confidence
        p = array1.argsort()
        array1 = array1[p]
        return_tuple = [array1, ]

        for array in args:
            array = array[p]
            return_tuple.append(array)

        return tuple(return_tuple)

    def _is_binary_classification(self) -> bool:
        """
        Determine if a given label vector is from a binary classification problem.
        This is important in order to process Histogram Binning.

        Returns
        -------
        bool
            True if binary classification problem (only two different labels occur), else False.
        """

        if self.num_classes == 2 and not self.independent_probabilities:
            return True
        else:
            return False

    @dimensions(1, None)
    def _get_one_vs_all_label(self, label_vec: np.ndarray, label: int) -> np.ndarray:
        """
        Return the all vs one vector for a given label vector. If an entry matches 'label', this entry
        is converted to 1, the rest to 0. The confidence for entries which are not 'label' are inverted (1 - conf).

        Parameters
        ----------
        label_vec : np.ndarray, shape=(n_samples,)
            NumPy 1-D array with label_vec (prediction or ground_truth).
        label : int
            Integer with label which will be replaced by 1 (rest by 0).

        Returns
        -------
        np.ndarray
            NumPy 1-D array with replaced labels ('label' by 1, rest by 0).
        """

        onevsall_label = np.array(label_vec, dtype=np.int64)
        onevsall_label[label_vec == label] = 1
        onevsall_label[label_vec != label] = 0

        return onevsall_label

    @dimensions(2, None)
    def _get_one_vs_all_confidence(self, confidences: np.ndarray, label: int) -> np.ndarray:
        """
        Return the all vs one vector for confidence. Each confidence entry will be calculated as (1 - conf)
        for each entry in 'onevsall_label' array which is 0.

        Parameters
        ----------
        confidences : np.ndarray, shape=(n_samples, n_classes)
            NumPy array with confidence values for each prediction.
            This must be 2-D for multi class (softmax).
        label : int
            Integer with label which will be replaced by 1 (rest by 0).

        Returns
        -------
        np.ndarray
            NumPy 1-D array with confidences for given label.
        """

        onevsall_conf = confidences[:, label]
        return onevsall_conf
