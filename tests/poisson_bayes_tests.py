# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
from riid.models.poisson_bayes import PoissonBayesClassifier
from riid.data import SampleSet
from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.static import StaticSynthesizer

import unittest
import numpy as np
import pandas as pd


class TestPoissonBayesClassifier(unittest.TestCase):
    """Test class for PoissonBayesClassifier class."""

    def test_constructor_and_predict(self):

        # Create the seed Sample Set
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
        bg_seeds_ss = seeds_ss[labels == BACKGROUND_LABEL]

        # Create the PoissonBayesClassifier
        pb_classifier = PoissonBayesClassifier(fg_seeds_ss)

        # Get test Sample Sets
        gss = StaticSynthesizer(
            samples_per_seed=1,
            live_time_function_args=(4, 4),
            snr_function_args=(10, 10),
            random_state=42
        )

        _, test_bg_ss, test_ss = gss.generate(
            fg_seeds_ss=fg_seeds_ss,  bg_seeds_ss=bg_seeds_ss)

        # Predict
        pb_classifier.predict(test_ss, test_bg_ss)

        truth_labels = fg_seeds_ss.get_labels()
        predictions_labels = test_ss.get_predictions()
        assert (truth_labels == predictions_labels).all()


if __name__ == '__main__':
    unittest.main()
