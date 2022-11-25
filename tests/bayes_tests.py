# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
import unittest

import numpy as np
import pandas as pd
from riid.data import SampleSet
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.bayes import (NegativeSpectrumError, PoissonBayes,
                               ZeroTotalCountsError)


class TestBayes(unittest.TestCase):
    """Test class for PoissonBayes class."""
    def setUp(self):
        """Test setup."""
        pass

    def test_constructor_errors(self):
        """Testing for constructor errors when different arguments are provided."""
        # No error when no argument provided
        _ = PoissonBayes()

        # Empty argument provided
        spectra = np.array([])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ValueError, PoissonBayes, ss)

        # Negative channel argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(NegativeSpectrumError, PoissonBayes, ss)

        # Zero total counts argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ZeroTotalCountsError, PoissonBayes, ss)

    def test_constructor_and_predict(self):
        """Tests the constructor with a valid SampleSet."""
        seed_spectra = np.array([
            [0.2, 0.2, 0.2, 0.4],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.2, 0.8],
            [0.1, 0.7, 0.2, 0.0],
        ])
        iso_seed_pairs = [
            ('SNM',     'U235',  'U235Unshielded'),
            ('NORM',    'K40',   'K40Unshielded'),
            ('SNM',     'Pu239', 'Pu239Unshielded'),
            (BACKGROUND_LABEL, BACKGROUND_LABEL, BACKGROUND_LABEL),
        ]
        source_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        sources = pd.DataFrame(
            data=source_matrix,
            columns=pd.MultiIndex.from_tuples(
                iso_seed_pairs,
                names=SampleSet.SOURCES_MULTI_INDEX_NAMES
            ),
        )

        seeds_ss = SampleSet()
        seeds_ss.spectra = pd.DataFrame(seed_spectra)
        seeds_ss.info.live_time = [1.0, 1.0, 1.0, 1.0]
        seeds_ss.info.snr = [0.0, 0.0, 0.0, 0.0]
        seeds_ss.info.gross_counts = [300.0, 300.0, 300.0, 300.0]
        seeds_ss.sources = sources

        labels = seeds_ss.get_labels()
        fg_seeds_ss = seeds_ss[labels != BACKGROUND_LABEL]
        fg_seeds_ss.sources.drop(BACKGROUND_LABEL, axis=1, level="Category", inplace=True)
        bg_seed_ss = seeds_ss[labels == BACKGROUND_LABEL]
        pb_model = PoissonBayes(fg_seeds_ss)

        static_syn = StaticSynthesizer(
            samples_per_seed=1,
            live_time_function_args=(4, 4),
            snr_function_args=(1, 1),
            random_state=42
        )
        _, _, test_gross_ss = static_syn.generate(fg_seeds_ss, bg_seed_ss, verbose=False)
        pb_model.predict_old(test_gross_ss, bg_seed_ss)
        test_gross_ss.sources.drop(BACKGROUND_LABEL, axis=1, level="Category", inplace=True)
        truths = test_gross_ss.get_labels()
        predictions = test_gross_ss.get_predictions(min_value=-1e4)
        assert (truths == predictions).all()


if __name__ == '__main__':
    unittest.main()
