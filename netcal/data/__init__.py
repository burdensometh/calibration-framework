# Copyright (C) 2019-2023 Ruhr West University of Applied Sciences, Bottrop, Germany
# AND e:fs TechHub GmbH, Gaimersheim, Germany
#
# This Source Code Form is subject to the terms of the Apache License 2.0
# If a copy of the APL2 was not distributed with this
# file, You can obtain one at https://www.apache.org/licenses/LICENSE-2.0.txt.

"""
.. include:: /../../netcal/data/README.md
   :parser: myst_parser.sphinx_

Available classes
=================

.. autosummary::
   :toctree: _autosummary_data
   :template: custom_class.rst

   CalibrationDataBuilder
   CalibrationDataset
   PredictionRecord
   LabelRecord
"""

from .CalibrationDataBuilder import (
    AsyncCalibrationDataBuilder,
    AsyncLabelError,
    CalibrationData,
    CalibrationDataBuilder,
    CalibrationDataset,
    EmptyCalibrationDataError,
    LabelConflictError,
    LabelRecord,
    MissingTimestampError,
    PredictionConflictError,
    PredictionRecord,
    build_calibration_data,
)
