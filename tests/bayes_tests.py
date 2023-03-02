# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
import unittest

import numpy as np
import pandas as pd
from riid.data import SampleSet
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer, get_dummy_seeds
from riid.models.bayes import (NegativeSpectrumError, PoissonBayesClassifier,
                               ZeroTotalCountsError)


class TestPoissonBayesClassifier(unittest.TestCase):
    """Test class for PoissonBayesClassifier."""
    def setUp(self):
        """Test setup."""
        pass

    def test_constructor_errors(self):
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

    def test_constructor_and_predict(self):
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
            random_state=42
        )
        _, test_bg_ss, test_ss = gss.generate(fg_seeds_ss, bg_seeds_ss, verbose=False)

        # Predict
        pb_model.predict(test_ss, test_bg_ss)

        truth_labels = fg_seeds_ss.get_labels()
        predictions_labels = test_ss.get_predictions()
        assert (truth_labels == predictions_labels).all()


if __name__ == '__main__':
    unittest.main()
