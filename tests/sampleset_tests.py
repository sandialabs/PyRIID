# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import unittest

import numpy as np
import pandas as pd

from riid.data.sampleset import SampleSet, _get_row_labels
from riid.data.synthetic.static import get_dummy_seeds


class TestSampleSet(unittest.TestCase):
    """Test class for SampleSet.
    """

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

    def test_as_squashed(self):
        """Tests that sampleset squashing sums data as expected."""
        ss = get_dummy_seeds()
        ss.info["snr_target"] = 0
        flat_ss = ss.as_squashed()

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
        self.assertEqual(flat_ss.info["snr_target"][0], ss.info["snr_target"].sum())
        self.assertEqual(flat_ss.info["snr"][0], ss.info["snr"].sum())
        self.assertEqual(flat_ss.info["sigma"][0], ss.info["sigma"].sum())
        self.assertEqual(flat_ss.info["bg_counts"][0], ss.info["bg_counts"].sum())
        self.assertEqual(flat_ss.info["fg_counts"][0], ss.info["fg_counts"].sum())
        self.assertEqual(flat_ss.info["bg_counts_expected"][0], ss.info["bg_counts_expected"].sum())
        self.assertEqual(flat_ss.info["fg_counts_expected"][0], ss.info["fg_counts_expected"].sum())
        self.assertEqual(flat_ss.info["gross_counts"][0], ss.info["gross_counts"].sum())
        self.assertEqual(flat_ss.info["gross_counts_expected"][0],
                         ss.info["gross_counts_expected"].sum())
        info_columns_are_the_same = np.array_equal(
            ss.info.columns,
            flat_ss.info.columns
        )
        self.assertTrue(info_columns_are_the_same)

    def test_get_row_labels_max_only_no_value_no_aggregation(self):
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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
        df = _get_test_df()
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

    def test_normalize_source(self):
        """Tests normalizing source matrix to valid probability distribution.
        """
        import sys
        np.set_printoptions(threshold=sys.maxsize)
        ss = get_dummy_seeds(n_channels=1024)
        ss.normalize_sources()

        sources = ss.sources.values
        self.assertTrue(np.allclose(sources.sum(axis=1), np.ones(sources.shape[0])))
        self.assertGreaterEqual(sources.min(), 0)

    def _assert_row_labels(self, level, actual, expected):
        for i, (a, e) in enumerate(zip(actual, expected)):
            self.assertEqual(
                a, e,
                f"Level '{level}', row '{i}' failure.  Actual '{a}' != Expected '{e}'"
            )


def _get_test_df():
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


if __name__ == '__main__':
    unittest.main()
