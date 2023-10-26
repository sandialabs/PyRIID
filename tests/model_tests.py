# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
import unittest

import numpy as np
import pandas as pd

from riid.data.sampleset import SampleSet
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.bayes import (NegativeSpectrumError, PoissonBayesClassifier,
                               ZeroTotalCountsError)
from riid.models.neural_nets import (LabelProportionEstimator, MLPClassifier,
                                     MultiEventClassifier)
from riid.models.neural_nets.arad import ARAD, ARADv1TF, ARADv2TF


class TestModels(unittest.TestCase):
    """Test class for PyRIID models."""
    def setUp(self):
        """Test setup."""
        pass

    def test_pb_constructor_errors(self):
        """Testing for constructor errors when different arguments are provided."""
        pb_model = PoissonBayesClassifier()

        # Empty argument provided
        spectra = np.array([])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ValueError, pb_model.fit, ss)

        # Negative channel argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(NegativeSpectrumError, pb_model.fit, ss)

        # Zero total counts argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ZeroTotalCountsError, pb_model.fit, ss)

    def test_pb_constructor_and_predict(self):
        """Tests the constructor with a valid SampleSet."""
        seeds_ss = get_dummy_seeds()
        fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()
        bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(1)

        # Create the PoissonBayesClassifier
        pb_model = PoissonBayesClassifier()
        pb_model.fit(fg_seeds_ss)

        # Get test samples
        gss = StaticSynthesizer(
            samples_per_seed=1,
            live_time_function_args=(4, 4),
            snr_function_args=(10, 10),
            rng=np.random.default_rng(42),
            return_fg=True,
            return_gross=True,
        )
        test_fg_ss, test_gross_ss = gss.generate(fg_seeds_ss, bg_seeds_ss, verbose=False)
        test_bg_ss = test_gross_ss - test_fg_ss

        # Predict
        pb_model.predict(test_gross_ss, test_bg_ss)

        truth_labels = fg_seeds_ss.get_labels()
        predictions_labels = test_gross_ss.get_predictions()
        assert (truth_labels == predictions_labels).all()

    def test_all_constructors(self):
        _ = PoissonBayesClassifier()
        _ = MLPClassifier()
        _ = LabelProportionEstimator()
        _ = MultiEventClassifier()
        arad_v1 = ARADv1TF()
        _ = ARAD(arad_v1)
        arad_v2 = ARADv2TF()
        _ = ARAD(arad_v2)


if __name__ == "__main__":
    unittest.main()
