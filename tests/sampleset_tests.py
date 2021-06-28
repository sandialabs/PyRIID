# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import copy
import unittest

import numpy as np
import pandas as pd
from riid.sampleset import MissingKeyForRelabeling, SampleSet


class TestSampleSet(unittest.TestCase):
    """Test class for SampleSet.
    """

    def setUp(self):
        """Test setup.
        """
        kwargs = {
            "measured_or_synthetic": "measured",
            "subtract_background": True,
            "purpose": "test",
            "comments": "TEST COMMENT STRING",
            "spectra": pd.DataFrame(
                data=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]],
            ),
            "sources": pd.DataFrame(
                data=[
                    [0, 1, 0, 0, "frogs"],
                    [1, 0, 0, 0, "cakes"],
                    [.1, 0, 0.9, 0, "turtles"],
                    [.1, 0, 0.0, 0.9, "dogs"]
                ],
                columns=["cakes", "cakes", "frogs in water", "dogs", "label"]
            )
        }
        self._ss = SampleSet(**kwargs)

    def test_constructor(self):
        """Testing the SampleSet constructor.
        """
        self.assertTrue(self._ss.measured_or_synthetic == "measured")
        self.assertTrue(self._ss.subtract_background)
        self.assertTrue(self._ss.purpose == "test")
        self.assertTrue(self._ss.comments == "TEST COMMENT STRING")

    def test_legacy_constructor_migration(self):
        """Testing the SampleSet constructor as it migrates an old SampleSet to the latest one.
        """
        pass

    def test_energy_conversion(self):
        """Tests conversion of spectra to energy bins.
        """
        spectra = self._ss.spectra.values
        self._ss.total_counts = spectra.sum(axis=1)
        self._ss.ecal_low_e = np.zeros(spectra.shape[0])
        self._ss.ecal_order_0 = 0
        self._ss.ecal_order_1 = 3000
        self._ss.ecal_order_2 = 100
        self._ss.ecal_order_3 = 0

        bins = np.linspace(0, 1, spectra.shape[1])
        energy_centers = bins * 3000 + bins**2 * 100
        self._ss.to_energy(energy_centers)
        for old, new in zip(spectra, self._ss.spectra.values):
            self.assertTrue(all(old == new), "Transform to original energy bins should not " +
                            "distort the spectrum.")

        energy_centers = np.array([3200, 6400, 8888, 9999])
        self._ss.to_energy(energy_centers)
        for new in zip(self._ss.spectra.values):
            self.assertTrue(all(new[0] == np.zeros(len(new))), "Transform to bins outside " +
                            "measured range should result in 0 counts for all channels.")

    def test_label_matrix_collapsing(self):
        """Tests that label matrix collapses redundant columns.
        """
        self.setUp()
        self.assertTrue(
            all(self._ss.sources.columns == ["cakes", "frogs in water", "dogs", "label"]),
            "Redundant columns were not collapsed on initialization."
        )
        value_check = np.all(
            self._ss.sources.values == np.array(
                [
                    [1.0, 0.0, 0.0, 'frogs'],
                    [1.0, 0.0, 0.0, 'cakes'],
                    [0.1, 0.9, 0.0, 'turtles'],
                    [0.1, 0.0, 0.9, 'dogs']
                ],
                dtype=object
            )
        )
        self.assertTrue(value_check, "Sources values were not combined correctly.")
        self._ss.label_matrix_labels = ["cakes", "frogs", "frogs"]
        self.assertTrue(
            all(self._ss.sources.columns == ["cakes", "frogs", "label"]),
            "Redundant columns were not collapsed on settings label_matrix_labels."
        )
        self.assertTrue(
            all(self._ss.label_matrix_labels == ["cakes", "frogs"]),
            "'label_matrix_labels' was incorrect."
        )

    def test_relabeling_single_isotope(self):
        """Tests that single isotope labels are updated.
        """
        self.setUp()
        orig_labels = copy.deepcopy(self._ss.labels)
        new_labels = orig_labels[::-1]
        self._ss.sources["label"] = new_labels
        self.assertTrue(all(new_labels == self._ss.labels), "Labels were not updated.")
        self._ss.labels = orig_labels
        self.assertTrue(all(orig_labels == self._ss.labels), "Labels were not updated.")

    def test_relabeling_from_dict(self):
        """Tests that relabeling from dictionary works for single, multi, and both
        """
        self.setUp()
        intial_single_labels = copy.deepcopy(self._ss.labels)
        intial_multi_labels = copy.deepcopy(self._ss.label_matrix_labels)
        single_dict = {}

        for i, label in enumerate(sorted(i for i in set(intial_single_labels))):
            single_dict.update({label: str(i)})

        combined_dict = copy.deepcopy(single_dict)
        multi_dict = {}
        for i, label in enumerate(set(intial_multi_labels)):
            multi_dict.update({label: str(i)})
            combined_dict.update({label: str(i)})

        single_expected = [single_dict[key] for key in intial_single_labels]
        multi_expected = [multi_dict[key] for key in intial_multi_labels]

        # Check for relabeling single labels
        self._ss.relabel_from_dict(single_dict, mode="single")
        self.assertTrue(
            all(self._ss.labels == single_expected),
            "Single relabel did not function as expected"
        )
        self.assertTrue(
            all(self._ss.label_matrix_labels == intial_multi_labels),
            "Single relabel modified multi labels."
        )

        # Check for relabeling label_matrix_labels
        self._ss.labels = intial_single_labels
        self._ss.relabel_from_dict(multi_dict, mode="multi")
        self.assertTrue(
            all(self._ss.labels == intial_single_labels),
            "Multi relabel modified single labels."
        )
        self.assertTrue(
            all(self._ss.label_matrix_labels == multi_expected),
            "Multi relable did not function as expected."
        )

        # Check that both can be relabeled simultaneously
        single_expected = [combined_dict[key] for key in intial_single_labels]
        multi_expected = [combined_dict[key] for key in intial_multi_labels]
        self._ss.labels = intial_single_labels
        self._ss.label_matrix_labels = intial_multi_labels
        self._ss.relabel_from_dict(combined_dict, mode="both")
        self.assertTrue(
            all(self._ss.labels.astype(int).astype(str) == single_expected),
            "All relabel did not result in expected single labels."
        )
        self.assertTrue(
            all(self._ss.label_matrix_labels == multi_expected),
            "All relabel did not result in expected multi labels."
        )

        # Check that missing keys raises correct exceptions
        self.setUp()
        with self.assertRaises(MissingKeyForRelabeling):
            self._ss.relabel_from_dict({}, mode="single")
        with self.assertRaises(MissingKeyForRelabeling):
            self._ss.relabel_from_dict({}, mode="multi")
        with self.assertRaises(MissingKeyForRelabeling):
            self._ss.relabel_from_dict({}, mode="both")

    def test_relabel_to_max_source(self):
        """Tests that single labels can be relabed to the maximum contributing source.
        """
        self.setUp()
        self._ss.relabel_to_max_source()
        expected_labels = ["cakes", "cakes", "frogs in water", "dogs"]
        self.assertTrue(
            all(self._ss.labels == expected_labels),
            "Labels were not relabed to maximum contribution label."
        )


if __name__ == '__main__':
    unittest.main()
