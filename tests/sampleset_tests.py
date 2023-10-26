# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import itertools
import unittest

import numpy as np
import pandas as pd

from riid.data.sampleset import (ChannelCountMismatchError,
                                 InvalidSampleCountError, SampleSet,
                                 SpectraState, SpectraStateMismatchError,
                                 SpectraType, _get_row_labels)
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer


class TestSampleSet(unittest.TestCase):
    """Test class for SampleSet.
    """

    def test__repr__(self):
        rng = np.random.default_rng(1)
        fg_seeds_ss, bg_seeds_ss = get_dummy_seeds(rng=rng)\
            .split_fg_and_bg()
        fg_ss, gross_ss = StaticSynthesizer(rng=rng)\
            .generate(fg_seeds_ss, bg_seeds_ss, verbose=False)
        fg_ss.__repr__()
        gross_ss.__repr__()

    def test__eq__(self):
        """Tests equality of two SampleSets"""
        ss1 = get_dummy_seeds(live_time=1, normalize=False)
        ss2 = get_dummy_seeds(live_time=5, normalize=False)

        self.assertTrue(ss1 == ss1)
        self.assertTrue(ss1 != ss2)

    def test_as_ecal(self):
        """Tests conversion of spectra to energy bins."""
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        ss = get_dummy_seeds(n_channels=1024)
        spectra = ss.spectra.values

        new_ss = ss.as_ecal(0, 3000, 100, 0, 0)

        self.assertTrue(
            np.allclose(spectra, new_ss.spectra.values),
            "Transform to original energy bins should not distort the spectrum."
        )

        new_ss = ss.as_ecal(4000, 6000, 0, 0, 0)
        for new in zip(new_ss.spectra.values):
            self.assertTrue(
                all(new[0] == np.zeros(len(new))),
                "Transform to bins outside measured range should result in 0 for all channels."
            )

    def test_squash(self):
        """Tests that sampleset squashing sums data as expected."""
        ss = get_dummy_seeds()
        ss.info["snr_target"] = 0
        flat_ss = ss.squash()

        self.assertEqual(flat_ss.n_samples, 1)
        self.assertEqual(ss.n_channels, flat_ss.n_channels)
        self.assertEqual(flat_ss.spectra.shape[0], 1)
        self.assertEqual(flat_ss.sources.shape[0], 1)
        self.assertEqual(flat_ss.prediction_probas.shape[0], 1)
        self.assertEqual(flat_ss.info.shape[0], 1)

        spectra_are_summed = np.array_equal(ss.spectra.sum(), flat_ss.spectra.sum())
        self.assertTrue(spectra_are_summed)

        sources_are_summed = np.array_equal(ss.sources.sum(), flat_ss.sources.sum())
        self.assertTrue(sources_are_summed)
        source_columns_are_the_same = np.array_equal(ss.sources.columns, flat_ss.sources.columns)
        self.assertTrue(source_columns_are_the_same)

        prediction_probas_are_summed = np.array_equal(
            ss.prediction_probas.sum(),
            flat_ss.prediction_probas.sum()
        )
        self.assertTrue(prediction_probas_are_summed)
        prediction_probas_columns_are_the_same = np.array_equal(
            ss.prediction_probas.columns,
            flat_ss.prediction_probas.columns
        )
        self.assertTrue(prediction_probas_columns_are_the_same)

        self.assertTrue(np.array_equal(flat_ss.ecal[0], ss.ecal[0]))
        self.assertEqual(flat_ss.info["description"][0], "squashed")
        self.assertEqual(flat_ss.info["timestamp"][0], ss.info["timestamp"][0])
        self.assertEqual(flat_ss.info["live_time"][0], ss.info["live_time"].sum())
        self.assertEqual(flat_ss.info["real_time"][0], ss.info["real_time"].sum())
        self.assertEqual(flat_ss.info["total_counts"][0], ss.info["total_counts"].sum())
        self.assertEqual(flat_ss.info["snr"][0], ss.info["snr"].sum())
        info_columns_are_the_same = np.array_equal(
            ss.info.columns,
            flat_ss.info.columns
        )
        self.assertTrue(info_columns_are_the_same)

    def test_check_arithmetic_supported(self):
        N_TARGET_CHANNELS = 5
        fg_ss = get_dummy_seeds(N_TARGET_CHANNELS)

        self.assertRaises(
            InvalidSampleCountError,
            fg_ss._check_arithmetic_supported,
            get_dummy_seeds(n_channels=N_TARGET_CHANNELS)[list(range(fg_ss.n_samples - 1))]
        )
        self.assertRaises(
            ChannelCountMismatchError,
            fg_ss._check_arithmetic_supported,
            get_dummy_seeds(n_channels=N_TARGET_CHANNELS + 1)[0]
        )
        mismatched_states = [
            (left, right)
            for left, right in itertools.combinations(SpectraState, r=2)
            if left != right
        ]
        for left_state, right_state in mismatched_states:
            l_ss = get_dummy_seeds(N_TARGET_CHANNELS)
            l_ss.spectra_state = left_state
            r_ss = get_dummy_seeds(N_TARGET_CHANNELS)[0]
            r_ss.spectra_state = right_state
            self.assertRaises(
                SpectraStateMismatchError,
                l_ss._check_arithmetic_supported,
                r_ss
            )
        unsupported_matching_states = [
            (left, right)
            for left, right in itertools.combinations(SpectraState, r=2)
            if left == right and left in SampleSet.SUPPORTED_STATES_FOR_ARITHMETIC
        ]
        for left_state, right_state in unsupported_matching_states:
            l_ss = get_dummy_seeds(N_TARGET_CHANNELS)
            l_ss.spectra_state = left_state
            r_ss = get_dummy_seeds(N_TARGET_CHANNELS)[0]
            r_ss.spectra_state = right_state
            self.assertRaises(
                ValueError,
                l_ss._check_arithmetic_supported,
                r_ss
            )

    def test_addition_and_subtraction_with_counts(self):
        rng = np.random.default_rng(42)
        default_fg_spectra = rng.integers(3, size=(10, 32))
        default_bg_spectra = rng.integers(3, size=(1, 32))
        ss1 = SampleSet()
        ss1.spectra_state = SpectraState.Counts
        ss1.spectra = pd.DataFrame(default_fg_spectra)
        ss1.info.total_counts = ss1.spectra.sum(axis=1)
        ss1.info.live_time = 1
        ss2 = SampleSet()
        ss2.spectra_state = SpectraState.Counts
        ss2.spectra = pd.DataFrame(default_bg_spectra)
        ss2.info.total_counts = ss1.spectra.sum(axis=1)
        ss2.info.live_time = 1

        ss1.spectra_type = SpectraType.Foreground
        ss2.spectra_type = SpectraType.Background

        ss3 = ss1 + ss2
        ss4 = ss3 - ss2

        self.assertTrue(ss1 == ss4)

    def test_addition_and_subtraction_with_l1_norm(self):
        rng = np.random.default_rng(42)
        default_fg_spectra = rng.integers(3, size=(10, 32))
        default_bg_spectra = rng.integers(3, size=(1, 32))
        ss1 = SampleSet()
        ss1.spectra = pd.DataFrame(default_fg_spectra)
        ss1.info.total_counts = ss1.spectra.sum(axis=1)
        ss1.info.live_time = 1
        ss1.normalize()
        ss2 = SampleSet()
        ss2.spectra = pd.DataFrame(default_bg_spectra)
        ss2.info.total_counts = ss1.spectra.sum(axis=1)
        ss2.info.live_time = 4
        ss2.normalize()

        ss1.spectra_type = SpectraType.Foreground
        ss2.spectra_type = SpectraType.Background

        ss3 = ss1 + ss2
        ss4 = ss3 - ss2

        self.assertTrue(ss1 == ss4)

    def test_get_row_labels_max_only_no_value_no_aggregation(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": False,
            "level_aggregation": None,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "X", "X", "Y",
            "Z"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A", "A", "A", "B",
            "C"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1", "A2", "A3", "B1",
            "C2"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_max_only_no_value_with_sum(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": False,
            "level_aggregation": "sum",
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "X", "X", "Y",
            "X"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A", "A", "A", "B",
            "B"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1", "A2", "A3", "B1",
            "C2"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_max_only_no_value_with_mean(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": False,
            "level_aggregation": "mean",
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "Z", "Z", "Y",
            "Y"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "D", "C", "C", "B",
            "B"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1", "A2", "A3", "B1",
            "C2"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_max_only_with_value_no_aggregation(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": True,
            "level_aggregation": None,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (0.50)", "X (0.50)", "X (0.50)", "Y (0.50)",
            "Z (0.25)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50)", "A (0.50)", "A (0.50)", "B (0.50)",
            "C (0.25)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50)", "A2 (0.50)", "A3 (0.50)", "B1 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_max_only_with_value_with_sum(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": True,
            "level_aggregation": "sum",
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (1.00)", "X (0.50)", "X (0.50)", "Y (1.00)",
            "X (0.40)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50)", "A (0.50)", "A (0.50)", "B (1.00)",
            "B (0.35)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50)", "A2 (0.50)", "A3 (0.50)", "B1 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_max_only_with_value_with_mean(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": True,
            "include_value": True,
            "level_aggregation": "mean",
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (0.25)", "X (0.25)", "X (0.25)", "Y (0.50)",
            "Y (0.50)", "Z (0.50)", "Z (0.50)", "X (0.25)",
            "X (0.25)", "Z (0.25)", "Z (0.25)", "Y (0.50)",
            "Y (0.17)"
        ]
        expected_isotopes = [
            "A (0.33)", "A (0.33)", "A (0.33)", "B (0.50)",
            "B (0.50)", "C (0.50)", "C (0.50)", "D (1.00)",
            "D (0.50)", "C (0.25)", "C (0.25)", "B (0.50)",
            "B (0.17)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50)", "A2 (0.50)", "A3 (0.50)", "B1 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_no_aggregation(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": None,
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X + X", "X + Z", "X + Z", "Y + Y",
            "X + X + X + Y + Y + Z + X"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A + D", "A + C", "A + C", "B + B",
            "A + A + A + B + B + C + D"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "A1 + A2 + A3 + B1 + B2 + C2 + D1"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_with_sum(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": "sum",
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "X + Z", "X + Z", "Y",
            "X + Y + Z"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A + D", "A + C", "A + C", "B",
            "A + B + C + D"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "A1 + A2 + A3 + B1 + B2 + C2 + D1"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_with_mean(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": "mean",
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "X + Z", "X + Z", "Y",
            "X + Y + Z"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A + D", "A + C", "A + C", "B",
            "A + B + C + D"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "A1 + A2 + A3 + B1 + B2 + C2 + D1"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_no_aggregation(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": None,
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (0.50) + X (0.50)", "X (0.50) + Z (0.50)",
            "X (0.50) + Z (0.50)", "Y (0.50) + Y (0.50)",
            "X (0.10) + X (0.10) + X (0.10) + Y (0.15) + Y (0.20) + Z (0.25) + X (0.10)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50) + D (0.50)", "A (0.50) + C (0.50)",
            "A (0.50) + C (0.50)", "B (0.50) + B (0.50)",
            "A (0.10) + A (0.10) + A (0.10) + B (0.15) + B (0.20) + C (0.25) + D (0.10)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "A1 (0.10) + A2 (0.10) + A3 (0.10) + B1 (0.15) + B2 (0.20) + C2 (0.25) + D1 (0.10)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_with_sum(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": "sum",
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (1.00)", "X (0.50) + Z (0.50)", "X (0.50) + Z (0.50)", "Y (1.00)",
            "X (0.40) + Y (0.35) + Z (0.25)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50) + D (0.50)", "A (0.50) + C (0.50)", "A (0.50) + C (0.50)", "B (1.00)",
            "A (0.30) + B (0.35) + C (0.25) + D (0.10)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "A1 (0.10) + A2 (0.10) + A3 (0.10) + B1 (0.15) + B2 (0.20) + C2 (0.25) + D1 (0.10)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_with_mean(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": "mean",
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (0.25)", "X (0.25)", "X (0.25)", "Y (0.50)",
            "Y (0.50)", "Z (0.50)", "Z (0.50)", "X (0.25)",
            "X (0.25)", "X (0.12) + Z (0.25)", "X (0.12) + Z (0.25)", "Y (0.50)",
            "X (0.10) + Y (0.17) + Z (0.12)"
        ]
        expected_isotopes = [
            "A (0.33)", "A (0.33)", "A (0.33)", "B (0.50)",
            "B (0.50)", "C (0.50)", "C (0.50)", "D (1.00)",
            "A (0.17) + D (0.50)", "A (0.17) + C (0.25)",
            "A (0.17) + C (0.25)", "B (0.50)",
            "A (0.10) + B (0.17) + C (0.12) + D (0.10)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "A1 (0.10) + A2 (0.10) + A3 (0.10) + B1 (0.15) + B2 (0.20) + C2 (0.25) + D1 (0.10)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_no_aggregation_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": None,
            "min_value": 0.25,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X + X", "X + Z", "X + Z", "Y + Y",
            "Z"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A + D", "A + C", "A + C", "B + B",
            "C"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "C2"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_with_sum_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": "sum",
            "min_value": 0.01,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "X + Z", "X + Z", "Y",
            "X + Y + Z"
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "A + D", "A + C", "A + C", "B",
            "A + B + C + D"
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "A1 + A2 + A3 + B1 + B2 + C2 + D1"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_no_value_with_mean_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": False,
            "level_aggregation": "mean",
            "min_value": 0.25,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X", "X", "X", "Y", "Y", "Z", "Z", "X",
            "X", "Z", "Z", "Y",
            ""
        ]
        expected_isotopes = [
            "A", "A", "A", "B", "B", "C", "C", "D",
            "D", "C", "C", "B",
            ""
        ]
        expected_seeds = [
            "A1", "A2", "A3", "B1", "B2", "C1", "C2", "D1",
            "A1 + D1", "A2 + C2", "A3 + C1", "B1 + B2",
            "C2"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_no_aggregation_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": None,
            "min_value": 0.25,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (0.50) + X (0.50)", "X (0.50) + Z (0.50)",
            "X (0.50) + Z (0.50)", "Y (0.50) + Y (0.50)",
            "Z (0.25)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50) + D (0.50)", "A (0.50) + C (0.50)",
            "A (0.50) + C (0.50)", "B (0.50) + B (0.50)",
            "C (0.25)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_with_sum_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": "sum",
            "min_value": 0.25,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (1.00)", "X (1.00)", "X (1.00)", "Y (1.00)",
            "Y (1.00)", "Z (1.00)", "Z (1.00)", "X (1.00)",
            "X (1.00)", "X (0.50) + Z (0.50)", "X (0.50) + Z (0.50)", "Y (1.00)",
            "X (0.40) + Y (0.35) + Z (0.25)"
        ]
        expected_isotopes = [
            "A (1.00)", "A (1.00)", "A (1.00)", "B (1.00)",
            "B (1.00)", "C (1.00)", "C (1.00)", "D (1.00)",
            "A (0.50) + D (0.50)", "A (0.50) + C (0.50)", "A (0.50) + C (0.50)", "B (1.00)",
            "A (0.30) + B (0.35) + C (0.25)"
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_get_row_labels_multiple_with_value_with_mean_with_min_value(self):
        df = _get_test_sources_df()
        test_kwargs = {
            "max_only": False,
            "include_value": True,
            "level_aggregation": "mean",
            "min_value": 0.25,
        }
        actual_categories = _get_row_labels(df, target_level="Category", **test_kwargs)
        actual_isotopes = _get_row_labels(df, target_level="Isotope", **test_kwargs)
        actual_seeds = _get_row_labels(df, target_level="Seed", **test_kwargs)

        expected_categories = [
            "X (0.25)", "X (0.25)", "X (0.25)", "Y (0.50)",
            "Y (0.50)", "Z (0.50)", "Z (0.50)", "X (0.25)",
            "X (0.25)", "Z (0.25)", "Z (0.25)", "Y (0.50)",
            ""
        ]
        expected_isotopes = [
            "A (0.33)", "A (0.33)", "A (0.33)", "B (0.50)",
            "B (0.50)", "C (0.50)", "C (0.50)", "D (1.00)",
            "D (0.50)", "C (0.25)",
            "C (0.25)", "B (0.50)",
            ""
        ]
        expected_seeds = [
            "A1 (1.00)", "A2 (1.00)", "A3 (1.00)", "B1 (1.00)",
            "B2 (1.00)", "C1 (1.00)", "C2 (1.00)", "D1 (1.00)",
            "A1 (0.50) + D1 (0.50)", "A2 (0.50) + C2 (0.50)",
            "A3 (0.50) + C1 (0.50)", "B1 (0.50) + B2 (0.50)",
            "C2 (0.25)"
        ]

        self._assert_row_labels("Category", actual_categories, expected_categories)
        self._assert_row_labels("Isotope", actual_isotopes, expected_isotopes)
        self._assert_row_labels("Seed", actual_seeds, expected_seeds)

    def test_normalize_sources(self):
        """Tests normalizing source matrix to valid probability distribution.
        """
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        ss = get_dummy_seeds(n_channels=1024)
        ss.normalize_sources()

        sources = ss.sources.values
        self.assertTrue(np.allclose(sources.sum(axis=1), np.ones(sources.shape[0])))
        self.assertGreaterEqual(sources.min(), 0)

    def test_compare_to(self):
        SYNTHETIC_DATA_CONFIG = {
            "samples_per_seed": 10,
            "bg_cps": 10,
            "snr_function": "uniform",
            "snr_function_args": (1, 100),
            "live_time_function": "uniform",
            "live_time_function_args": (0.25, 10),
            "apply_poisson_noise": True,
            "return_fg": False,
            "return_gross": True,
        }
        fg_seeds_ss1, bg_seeds_ss1 = get_dummy_seeds().split_fg_and_bg()
        static_syn1 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)
        _, gross_ss1 = static_syn1.generate(fg_seeds_ss1, bg_seeds_ss1, verbose=False)

        fg_seeds_ss2, bg_seeds_ss2 = get_dummy_seeds().split_fg_and_bg()
        static_syn2 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)
        _, gross_ss2 = static_syn2.generate(fg_seeds_ss2, bg_seeds_ss2, verbose=False)

        _, _, _ = gross_ss1.compare_to(gross_ss2)
        _, _, col_comparison_same = gross_ss1.compare_to(gross_ss1)
        for k, v in col_comparison_same.items():
            self.assertTrue(
                v == 0.0,
                f"Key '{k}' failed to be zero when comparing SampleSet to itself."
            )

    def test_spectral_distance(self):
        seeds_ss = get_dummy_seeds()
        distance_df = seeds_ss.get_spectral_distance_matrix()
        upper_indices = np.triu_indices(
            n=distance_df.shape[0],
            k=1,
            m=distance_df.shape[1],
        )
        lower_indices = np.tril_indices(
            n=distance_df.shape[0],
            k=-1,
            m=distance_df.shape[1],
        )
        diagonal_indices = np.diag_indices(
            distance_df.values.shape[0]
        )

        diagonal_values = distance_df.values[diagonal_indices]
        upper_triangle_values = distance_df.values[upper_indices]
        lower_triangle_values = distance_df.values[lower_indices]

        self.assertTrue(all(diagonal_values == 0.0))
        self.assertTrue(all(lower_triangle_values == 0.0))
        self.assertTrue(all(upper_triangle_values > 0.0))

    def test_get_confidences(self):
        random_state = 42
        rng = np.random.default_rng(random_state)
        seeds_ss = get_dummy_seeds(rng=rng, n_channels=256)
        fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()

        fg_seeds_ss.prediction_probas = pd.DataFrame(
            data=fg_seeds_ss.sources.values,
            columns=fg_seeds_ss.sources.columns
        )
        perfect_fg_conf = fg_seeds_ss.get_confidences(fg_seeds_ss[:])

        mixer = SeedMixer(fg_seeds_ss, mixture_size=3, rng=rng)
        mixed_fg_seeds_ss = mixer.generate(50)
        mixed_fg_seeds_ss.prediction_probas = pd.DataFrame(
            data=mixed_fg_seeds_ss.sources.values,
            columns=mixed_fg_seeds_ss.sources.columns
        )
        perfect_mixed_fg_conf = mixed_fg_seeds_ss.get_confidences(fg_seeds_ss[:], is_lpe=True)

        synth = StaticSynthesizer(
            samples_per_seed=1,
            apply_poisson_noise=False,
            return_fg=False,
            return_gross=True,
            rng=rng
        )
        _, synthetic_gross_ss = synth.generate(fg_seeds_ss, bg_seeds_ss[0])
        synthetic_gross_ss.drop_sources(bg_seeds_ss.sources.columns.levels[2])
        synthetic_gross_ss.sources = synthetic_gross_ss.sources[fg_seeds_ss.sources.columns]
        synthetic_gross_ss.prediction_probas = pd.DataFrame(
            data=synthetic_gross_ss.sources.values,
            columns=synthetic_gross_ss.sources.columns
        )
        perfect_gross_conf = synthetic_gross_ss.get_confidences(
            fg_seeds_ss[:],
            bg_seed_ss=bg_seeds_ss[0],
            bg_cps=synth.bg_cps
        )

        _, synthetic_mixed_gross_ss = synth.generate(mixed_fg_seeds_ss, bg_seeds_ss[0])
        synthetic_mixed_gross_ss.drop_sources(bg_seeds_ss.sources.columns.levels[2])
        synthetic_mixed_gross_ss.sources = synthetic_mixed_gross_ss.sources[
            fg_seeds_ss.sources.columns
        ]
        synthetic_mixed_gross_ss.prediction_probas = pd.DataFrame(
            data=synthetic_mixed_gross_ss.sources.values,
            columns=synthetic_mixed_gross_ss.sources.columns
        )
        perfect_mixed_gross_conf = synthetic_mixed_gross_ss.get_confidences(
            fg_seeds_ss[:],
            bg_seed_ss=bg_seeds_ss[0],
            bg_cps=synth.bg_cps,
            is_lpe=True
        )

        self.assertTrue(np.all(np.isclose(perfect_fg_conf, 0.0)))
        self.assertTrue(np.all(np.isclose(perfect_mixed_fg_conf, 0.0)))
        self.assertTrue(np.all(np.isclose(perfect_gross_conf, 0.0)))
        self.assertTrue(np.all(np.isclose(perfect_mixed_gross_conf, 0.0)))

        with self.assertRaises(ValueError):
            fg_seeds_ss_wrong_type = fg_seeds_ss[:]
            fg_seeds_ss_wrong_type.spectra_type = SpectraType.Gross
            fg_seeds_ss.get_confidences(fg_seeds_ss_wrong_type[:])
        with self.assertRaises(ValueError):
            synthetic_gross_ss_empty_spectrum = synthetic_gross_ss[:]
            synthetic_gross_ss_empty_spectrum.info.loc[0, "total_counts"] = 0
            synthetic_gross_ss_empty_spectrum.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=bg_seeds_ss[0],
                bg_cps=synth.bg_cps
            )
        with self.assertRaises(ValueError):
            synthetic_gross_ss_zero_lt = synthetic_gross_ss[:]
            synthetic_gross_ss_zero_lt.info.loc[0, "live_time"] = 0
            synthetic_gross_ss_zero_lt.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=bg_seeds_ss[0],
                bg_cps=synth.bg_cps
            )
        with self.assertRaises(ValueError):
            synthetic_gross_ss.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=None,
                bg_cps=synth.bg_cps
            )
        with self.assertRaises(ValueError):
            synthetic_gross_ss.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=bg_seeds_ss[0:2],
                bg_cps=synth.bg_cps
            )
        with self.assertRaises(ValueError):
            bg_seeds_ss_wrong_type = bg_seeds_ss[:]
            bg_seeds_ss_wrong_type.spectra_type = SpectraType.Foreground
            synthetic_gross_ss.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=bg_seeds_ss_wrong_type[0],
                bg_cps=synth.bg_cps
            )
        with self.assertRaises(ValueError):
            synthetic_gross_ss.get_confidences(
                fg_seeds_ss[:],
                bg_seed_ss=bg_seeds_ss[0],
                bg_cps=None
            )

    def _assert_row_labels(self, level, actual, expected):
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertEqual(
                a, e,
                f"Level '{level}', row '{i}' failure.  Actual '{a}' != Expected '{e}'"
            )


def _get_test_sources_df():
    df = pd.DataFrame(
        [
            # Single-isotope
            (1, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 1),
            # Multi-isotope
            # Multiple answers of the same value - first is correct
            (0.5,   0,   0,   0,   0,   0,   0, 0.5),
            (0,   0.5,   0,   0,   0,   0, 0.5,   0),
            (0,     0, 0.5,   0,   0, 0.5,   0,   0),
            (0,     0,   0, 0.5, 0.5,   0,   0,   0),
            # Answer changes depending on target level and aggregation
            (0.1, 0.1, 0.1, 0.15, 0.2, 0, 0.25, 0.1),
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("X", "A", "A1"),
                ("X", "A", "A2"),
                ("X", "A", "A3"),
                ("Y", "B", "B1"),
                ("Y", "B", "B2"),
                ("Z", "C", "C1"),
                ("Z", "C", "C2"),
                ("X", "D", "D1"),
            ],
            names=SampleSet.SOURCES_MULTI_INDEX_NAMES
        )
    )
    return df


if __name__ == "__main__":
    unittest.main()
