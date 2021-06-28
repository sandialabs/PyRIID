# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
import unittest

import numpy as np
import pandas as pd
from riid.models.bayes import (NegativeSpectrumError, PoissonBayes,
                                     ZeroTotalCountsError)
from riid.sampleset import SampleSet
from riid.synthetic import GammaSpectraSynthesizer


class TestBayes(unittest.TestCase):
    """Test class for PoissonBayes class.
    """
    def setUp(self):
        """Test setup.
        """
        pass

    def test_constructor_errors(self):
        """Testing for constructor errors when different arguments are provided.
        """
        # No error when no argument provided
        _ = PoissonBayes()

        # Empty argument provided
        spectra = np.array([])
        ss = SampleSet(spectra=pd.DataFrame(spectra))
        self.assertRaises(ValueError, PoissonBayes, ss)

        # Negative channel argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1]
        ])
        ss = SampleSet(spectra=pd.DataFrame(spectra))
        self.assertRaises(NegativeSpectrumError, PoissonBayes, ss)

        # Zero total counts argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        ss = SampleSet(spectra=pd.DataFrame(spectra))
        self.assertRaises(ZeroTotalCountsError, PoissonBayes, ss)

    def test_constructor_and_predict(self):
        """Tests the constructor with a valid SampleSet.
        """
        seed_spectra = np.array([
            [0.3, 0.3, 0.2, 0.4],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.2, 0.8],
            [0.1, 0.7, 0.2, 0.0],
        ])
        isotopes = ["background", "a", "b", "c"]
        source_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        sources = pd.DataFrame(
            columns=isotopes,
            data=source_matrix
        )
        sources["label"] = [isotopes[i] for i in source_matrix.argmax(axis=1)]

        seeds_ss = SampleSet(
            purpose="seeds",
            spectra=pd.DataFrame(seed_spectra),
            collection_information=pd.DataFrame.from_dict({
                "live_time": [1.0, 1.0, 1.0, 1.0],
                "snr_estimate": [0.0, 0.0, 0.0, 0.0],
                "total_counts": [300.0, 300.0, 300.0, 300.0],
            }),
            sources=sources
        )
        pb_model = PoissonBayes(seeds_ss)

        gss = GammaSpectraSynthesizer(
            seeds_ss,
            purpose="test",
            samples_per_seed=1,
            subtract_background=False,
            live_time_function_args=(4, 4),
            snr_function_args=(15, 15),
            random_state=42
        )
        test_ss = gss.generate()
        bg_seed_ss = seeds_ss.get_indices(seeds_ss.labels == "background")
        pb_model.predict(test_ss, bg_seed_ss)
        assert set(test_ss.predictions) == set(test_ss.labels)


if __name__ == '__main__':
    unittest.main()
