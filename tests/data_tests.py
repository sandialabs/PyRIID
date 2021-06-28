# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module tests the data module."""
import os
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from riid import DataDirectoryNotFoundError
from riid.data import _check_data_path, load_samples_from_file, write_hdf
from riid.sampleset import SampleSet


class TestData(unittest.TestCase):
    """Test class for Data module."""

    def setUp(self):
        """Test setup."""
        kwargs = {
            "measured_or_synthetic": "measured",
            "subtract_background": True,
            "purpose": "test",
            "comments": "TEST COMMENT STRING"
        }
        self._sample_set = SampleSet(**kwargs)

        # Save as pickle and as hdf here
        # Save sampleset missing various collection info columns

    def test_load_pickle_samples(self):
        """Tests loading of legacy samplesets saved in pickle format.
        """
        test_file_path = "test.smpl"
        sampleset = SampleSet(**default_sampleset_params())

        with open(test_file_path, "wb") as fout:
            pickle.dump(sampleset, fout)

        sampleset_out = load_samples_from_file(test_file_path, False)

        compare_samplesets(self, sampleset, sampleset_out)
        os.remove(test_file_path)

    def test_load_hdf_samples(self):
        """Tests loading of samplesets saved in hdf format.
        """
        test_file_path = "test.smpl"
        sampleset = SampleSet(**default_sampleset_params())
        write_hdf(sampleset, test_file_path)
        sampleset_out = load_samples_from_file(test_file_path, verbose=0)
        compare_samplesets(self, sampleset, sampleset_out)
        os.remove(test_file_path)

    def test_load_pickle_samples_with_missing_collection_info(self):
        """Tests loading of legacy samplesets saved in pickle format with missing info."""
        test_file_path = "test.smpl"
        params = default_sampleset_params()

        # Drop several columns of the collection information DataFrame
        df_columns = params["collection_information"].columns
        params["collection_information"] = params["collection_information"].drop(
                np.random.choice(df_columns, 4, replace=False),
                axis=1
        )
        sampleset = SampleSet(**params)

        with open(test_file_path, "wb") as fout:
            pickle.dump(sampleset, fout)

        sampleset = load_samples_from_file(test_file_path, False)
        self.assertTrue(
            all(column in sampleset.collection_information.columns for column in df_columns),
            "Expect all columns to be present in collection information after loading."
        )
        os.remove(test_file_path)

    def test_load_hdf_samples_with_missing_collection_info(self):
        """Tests loading of legacy samplesets saved in hdf format with missing info."""
        test_file_path = "test.smpl"
        params = default_sampleset_params()

        # Drop several columns of the collection information DataFrame
        df_columns = params["collection_information"].columns
        params["collection_information"] = params["collection_information"].drop(
            np.random.choice(df_columns, 4, replace=False),
            axis=1
        )
        sampleset = SampleSet(**params)
        write_hdf(sampleset, test_file_path)

        sampleset = load_samples_from_file(test_file_path, False)
        self.assertTrue(
            all(column in sampleset.collection_information.columns for column in df_columns),
            "Expect all columns to be present in collection information after loading."
        )
        os.remove(test_file_path)

    def test_check_data_path(self):
        """Tests that checking for the data path is correct."""
        # Exception when it doesn't exist
        def path_doesnt_exists(arg):
            return False
        patcher = patch("os.path.exists")
        mock_thing = patcher.start()
        mock_thing.side_effect = path_doesnt_exists
        self.assertRaises(DataDirectoryNotFoundError, lambda: _check_data_path())

        # No exception when it does exist
        def path_exists(arg):
            return True
        mock_thing.side_effect = path_exists
        _check_data_path()
        patcher.stop()


def compare_samplesets(unit_test, ss1: SampleSet, ss2: SampleSet):
    """Compares two samplesets
    """
    msg = "Expect contents to be unchanged storage and retrieval."
    for v1, v2 in zip(sorted(ss1.__dict__), sorted(ss2.__dict__)):
        unit_test.assertEqual(v1, v2, "objects must have the same elements")
        if isinstance(ss1[v1], pd.DataFrame):
            if ss1[v1].empty:
                unit_test.assertEqual(ss1[v1].empty, ss2[v2].empty, msg)
            else:
                comp = [i for i in (a == b for a, b in zip(ss1[v1].values, ss2[v2].values))]
                unit_test.assertTrue(np.alltrue(comp), msg)
        elif isinstance(ss1[v1], np.ndarray) and isinstance(ss2[v2], np.ndarray):
            if ss1[v1].shape[0]:
                unit_test.assertTrue(np.alltrue(ss1[v1] == ss2[v2]), msg)
            else:
                unit_test.assertEqual(ss1[v1].shape, ss2[v2].shape, msg)
        else:
            unit_test.assertEqual(ss1[v1], ss2[v2], msg)


def default_sampleset_params():
    """ Gets default sampleset parameters.
    """
    df_columns = [
        "live_time",
        "snr_target",
        "snr_estimate",
        "bg_counts",
        "fg_counts",
        "bg_counts_expected",
        "total_counts",
        "sigma",
        "distance",
        "atomic_number",
        "area_density",
        "ecal_order_0",
        "ecal_order_1",
        "ecal_order_2",
        "ecal_order_3",
        "ecal_low_e",
        "date-time",
        "real_time",
        "occupancy_flag",
        "tag",
        "total_neutron_counts",
        "descr"
    ]

    collection_information = {
        key: value for key, value in zip(df_columns, np.random.rand(len(df_columns)))
    }
    params = {
        "collection_information": pd.DataFrame(data=[collection_information]),
        "sensor_information": {},
        "config": None,
        "prediction_probas": pd.DataFrame(np.array([])),
        "predictions": np.array([]),
        "measured_or_synthetic": None,
        "subtract_background": None,
        "purpose": "test",
        "comments": "Comments Text Here",
        "energy_bin_centers": np.array([]),
        "energy_bin_edges": np.array([]),
        "spectra": pd.DataFrame(data=np.random.randint(0, 1000, [1, 1024]))
    }
    return params


if __name__ == '__main__':
    unittest.main()
