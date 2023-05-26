# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the gadras module."""
import unittest
import yaml
import copy

from riid.gadras.api import (
    validate_inject_config,
    get_expanded_config,
    get_detector_setups,
    get_inject_setups,
)


base_config_yml = """
---
random_seed: 42
gamma_detector:
  name: Generic\\NaI\\2x4x16
  parameters:
    distance_cm: 1000
    height_cm: 45
    dead_time_per_pulse: 5
    latitude_deg: 35.0
    longitude_deg: 253.4
    elevation_m: 1620
sources:
  - isotope: Am241
    configurations:
      - Am241,100.5uCi
"""
base_config = yaml.safe_load(base_config_yml)


class TestConfig(unittest.TestCase):
    """Helper function"""

    def return_validate_inject_config(self, config):
        try:
            validate_inject_config(config)
            return True
        except Exception as e:
            raise Exception(str(e))
        return True

    "************************************************************************"
    "Tests for Random Seed"
    "************************************************************************"

    def test_valid_random_seed(self):
        # Modify base config for this test
        valid_random_seed = copy.deepcopy(base_config)

        """Tests config validation."""
        assert self.return_validate_inject_config(valid_random_seed)

    def test_invalid_random_seed(self):
        # Modify base config for this test
        valid_invalid_random_seed = copy.deepcopy(base_config)
        valid_invalid_random_seed["random_seed"] = 42.1

        """Tests config validation."""
        self.assertRaises(
            Exception, self.return_validate_inject_config, valid_invalid_random_seed
        )

    "************************************************************************"
    "Tests for Detector Configs"
    "************************************************************************"

    def test_valid_validate_inject_config_detector_single_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_a_single_number = copy.deepcopy(base_config)

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_a_single_number
        )

    def test_valid_validate_inject_config_detector_list_single_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_of_single_number = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [1000]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["height_cm"] = [45]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [5]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [35.0]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [253.4]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [1620]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_list_of_single_number
        )

    def test_valid_validate_inject_config_detector_list_mult_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_of_mult_number = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [1000, 2000]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["height_cm"] = [45, 46]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [5, 6]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [35.0, 36.0]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [253.4, 254.4]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [1620, 1621]
        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_list_of_mult_number
        )

    def test_valid_validate_inject_config_detector_single_sample_norm(self):
        # Modify base config for this test
        valid_config_detector_with_distance_single_sample_norm = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["distance_cm"] = {"mean": 1000, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["height_cm"] = {"mean": 45, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = {"mean": 5, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = {"mean": 35.0, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = {"mean": 253.4, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["elevation_m"] = {"mean": 1620, "std": 1, "num_samples": 1}

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_single_sample_norm
        )

    def test_valid_validate_inject_config_detector_list_sample_norm(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_sample_norm = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [{"mean": 1000, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["height_cm"] = [{"mean": 45, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [{"mean": 5, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [{"mean": 35.0, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [{"mean": 253.4, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [{"mean": 1620, "std": 1, "num_samples": 1}]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_list_sample_norm
        )

    def test_valid_validate_inject_config_detector_single_sample_range(self):
        # Modify base config for this test
        valid_config_detector_with_distance_single_sample_range = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["distance_cm"] = {
            "min": 1000,
            "max": 2000,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["height_cm"] = {"min": 45, "max": 55, "dist": "uniform", "num_samples": 1}
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = {
            "min": 5,
            "max": 10,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = {
            "min": 35.0,
            "max": 45.0,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = {
            "min": 253.4,
            "max": 263.4,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["elevation_m"] = {
            "min": 1620,
            "max": 1720,
            "dist": "uniform",
            "num_samples": 1,
        }

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_single_sample_range
        )

    def test_valid_validate_inject_config_detector_list_sample_range(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_sample_range = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [
            {"min": 1000, "max": 2000, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["height_cm"] = [{"min": 45, "max": 55, "dist": "uniform", "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [
            {"min": 5, "max": 10, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [
            {"min": 35.0, "max": 45.0, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [
            {"min": 253.4, "max": 263.4, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [
            {"min": 1620, "max": 1720, "dist": "uniform", "num_samples": 1}
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_list_sample_range
        )

    def test_valid_validate_inject_config_detector_list_all(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_all = copy.deepcopy(base_config)
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "distance_cm"
        ] = [
            1000,
            2000,
            {"mean": 1, "std": 1, "num_samples": 1},
            {"min": 1, "max": 10, "dist": "uniform", "num_samples": 1},
            3000,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "height_cm"
        ] = [
            45,
            55,
            {"mean": 45, "std": 1, "num_samples": 1},
            {"min": 45, "max": 55, "dist": "uniform", "num_samples": 1},
            65,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "dead_time_per_pulse"
        ] = [
            5,
            10,
            {"mean": 5, "std": 1, "num_samples": 1},
            {"min": 5, "max": 10, "dist": "uniform", "num_samples": 1},
            15,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "latitude_deg"
        ] = [
            35.0,
            45.0,
            {"mean": 35.0, "std": 1, "num_samples": 1},
            {"min": 35.0, "max": 45.0, "dist": "uniform", "num_samples": 1},
            55.0,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "longitude_deg"
        ] = [
            253.4,
            263.4,
            {"mean": 253.4, "std": 1, "num_samples": 1},
            {"min": 253.4, "max": 263.4, "dist": "uniform", "num_samples": 1},
            273.4,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "elevation_m"
        ] = [
            1620,
            1720,
            {"mean": 1620, "std": 1, "num_samples": 1},
            {"min": 1620, "max": 1720, "dist": "uniform", "num_samples": 1},
            1820,
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_detector_with_distance_list_all
        )

    def test_invalid_validate_inject_config(self):
        # Modify base config for this test
        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"][
            "distance_cm"
        ] = "a_string_which_is_an_error"
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"]["height_cm"] = True
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"][
            "dead_time_per_pulse"
        ] = "a_string_which_is_an_error"
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"]["latitude_deg"] = False
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"][
            "longitude_deg"
        ] = "a_string_which_is_an_error"
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

        invalid_config = copy.deepcopy(base_config)
        invalid_config["gamma_detector"]["parameters"]["elevation_m"] = True
        self.assertRaises(Exception, self.return_validate_inject_config, invalid_config)

    # Expanded Configs
    def test_get_expanded_config_detector_single_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_a_single_number = copy.deepcopy(base_config)

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_a_single_number
        )

        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_list_single_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_of_single_number = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [1000]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["height_cm"] = [45]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [5]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [35.0]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [253.4]
        valid_config_detector_with_distance_list_of_single_number["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [1620]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_list_of_single_number
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_list_mult_number(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_of_mult_number = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [1000, 2000]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["height_cm"] = [45, 46]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [5, 6]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [35.0, 36.0]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [253.4, 254.4]
        valid_config_detector_with_distance_list_of_mult_number["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [1620, 1621]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000, 2000],
                    "height_cm": [45, 46],
                    "dead_time_per_pulse": [5, 6],
                    "latitude_deg": [35.0, 36.0],
                    "longitude_deg": [253.4, 254.4],
                    "elevation_m": [1620, 1621],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_list_of_mult_number
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_single_sample_norm(self):
        # Modify base config for this test
        valid_config_detector_with_distance_single_sample_norm = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_single_sample_norm["random_seed"] = 1
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["distance_cm"] = {"mean": 1000, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["height_cm"] = {"mean": 45, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = {"mean": 5, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = {"mean": 35.0, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = {"mean": 253.4, "std": 1, "num_samples": 1}
        valid_config_detector_with_distance_single_sample_norm["gamma_detector"][
            "parameters"
        ]["elevation_m"] = {"mean": 1620, "std": 1, "num_samples": 1}

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000.3455841920647],
                    "height_cm": [45.821618143501155],
                    "dead_time_per_pulse": [5.3304370761833875],
                    "latitude_deg": [33.69684276839564],
                    "longitude_deg": [254.30535586667312],
                    "elevation_m": [1620.446374572364],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_single_sample_norm
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_list_sample_norm(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_sample_norm = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_sample_norm["random_seed"] = 1
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [{"mean": 1000, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["height_cm"] = [{"mean": 45, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [{"mean": 5, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [{"mean": 35.0, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [{"mean": 253.4, "std": 1, "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_norm["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [{"mean": 1620, "std": 1, "num_samples": 1}]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000.3455841920647],
                    "height_cm": [45.821618143501155],
                    "dead_time_per_pulse": [5.3304370761833875],
                    "latitude_deg": [33.69684276839564],
                    "longitude_deg": [254.30535586667312],
                    "elevation_m": [1620.446374572364],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_list_sample_norm
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_single_sample_range(self):
        # Modify base config for this test
        valid_config_detector_with_distance_single_sample_range = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_single_sample_range["random_seed"] = 1
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["distance_cm"] = {
            "min": 1000,
            "max": 2000,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["height_cm"] = {"min": 45, "max": 55, "dist": "uniform", "num_samples": 1}
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = {
            "min": 5,
            "max": 10,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = {
            "min": 35.0,
            "max": 45.0,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = {
            "min": 253.4,
            "max": 263.4,
            "dist": "uniform",
            "num_samples": 1,
        }
        valid_config_detector_with_distance_single_sample_range["gamma_detector"][
            "parameters"
        ]["elevation_m"] = {
            "min": 1620,
            "max": 1720,
            "dist": "uniform",
            "num_samples": 1,
        }

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1511.8216247002567],
                    "height_cm": [54.50463696325935],
                    "dead_time_per_pulse": [5.720798063598169],
                    "latitude_deg": [44.48649447137244],
                    "longitude_deg": [256.51831452010487],
                    "elevation_m": [1662.3326448972575],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_single_sample_range
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_list_sample_range(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_sample_range = copy.deepcopy(
            base_config
        )
        valid_config_detector_with_distance_list_sample_range["random_seed"] = 1
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["distance_cm"] = [
            {"min": 1000, "max": 2000, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["height_cm"] = [{"min": 45, "max": 55, "dist": "uniform", "num_samples": 1}]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["dead_time_per_pulse"] = [
            {"min": 5, "max": 10, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["latitude_deg"] = [
            {"min": 35.0, "max": 45.0, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["longitude_deg"] = [
            {"min": 253.4, "max": 263.4, "dist": "uniform", "num_samples": 1}
        ]
        valid_config_detector_with_distance_list_sample_range["gamma_detector"][
            "parameters"
        ]["elevation_m"] = [
            {"min": 1620, "max": 1720, "dist": "uniform", "num_samples": 1}
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1511.8216247002567],
                    "height_cm": [54.50463696325935],
                    "dead_time_per_pulse": [5.720798063598169],
                    "latitude_deg": [44.48649447137244],
                    "longitude_deg": [256.51831452010487],
                    "elevation_m": [1662.3326448972575],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_detector_with_distance_list_sample_range
        )
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_detector_list_all(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_all = copy.deepcopy(base_config)
        valid_config_detector_with_distance_list_all["random_seed"] = 1
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "distance_cm"
        ] = [
            1000,
            2000,
            {"mean": 1, "std": 1, "num_samples": 1},
            {"min": 1, "max": 10, "dist": "uniform", "num_samples": 1},
            3000,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "height_cm"
        ] = [
            45,
            55,
            {"mean": 45, "std": 1, "num_samples": 1},
            {"min": 45, "max": 55, "dist": "uniform", "num_samples": 1},
            65,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "dead_time_per_pulse"
        ] = [
            5,
            10,
            {"mean": 5, "std": 1, "num_samples": 1},
            {"min": 5, "max": 10, "dist": "uniform", "num_samples": 1},
            15,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "latitude_deg"
        ] = [
            35.0,
            45.0,
            {"mean": 35.0, "std": 1, "num_samples": 1},
            {"min": 35.0, "max": 45.0, "dist": "uniform", "num_samples": 1},
            55.0,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "longitude_deg"
        ] = [
            253.4,
            263.4,
            {"mean": 253.4, "std": 1, "num_samples": 1},
            {"min": 253.4, "max": 263.4, "dist": "uniform", "num_samples": 1},
            273.4,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "elevation_m"
        ] = [
            1620,
            1720,
            {"mean": 1620, "std": 1, "num_samples": 1},
            {"min": 1620, "max": 1720, "dist": "uniform", "num_samples": 1},
            1820,
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [
                        1000,
                        2000,
                        1.345584192064786,
                        9.554173266933418,
                        3000,
                    ],
                    "height_cm": [45, 55, 45.33043707618339, 54.48649447137244, 65],
                    "dead_time_per_pulse": [
                        5,
                        10,
                        5.905355866673117,
                        7.1166322448628785,
                        15,
                    ],
                    "latitude_deg": [
                        35.0,
                        45.0,
                        34.463046764639714,
                        39.09199136369161,
                        55.0,
                    ],
                    "longitude_deg": [
                        253.4,
                        263.4,
                        253.76457239618608,
                        253.6755911324307,
                        273.4,
                    ],
                    "elevation_m": [
                        1620,
                        1720,
                        1620.0284222413159,
                        1673.8143313219277,
                        1820,
                    ],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(valid_config_detector_with_distance_list_all)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_detector_setups_list_all(self):
        # Modify base config for this test
        valid_config_detector_with_distance_list_all = copy.deepcopy(base_config)
        valid_config_detector_with_distance_list_all["random_seed"] = 1
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "distance_cm"
        ] = [
            1000,
            2000,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "height_cm"
        ] = [
            45,
            55,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "dead_time_per_pulse"
        ] = [
            5,
            10,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "latitude_deg"
        ] = [
            35.0,
            45.0,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "longitude_deg"
        ] = [
            253.4,
            263.4,
        ]
        valid_config_detector_with_distance_list_all["gamma_detector"]["parameters"][
            "elevation_m"
        ] = [
            1620,
            1720,
        ]
        expanded_config = get_expanded_config(
            valid_config_detector_with_distance_list_all
        )
        detector_configs = get_detector_setups(expanded_config)

        # Asserts
        assert len(detector_configs) == 64
        assert detector_configs[0]["parameters"]["distance_cm"] == 1000
        assert detector_configs[32]["parameters"]["distance_cm"] == 2000

        assert detector_configs[0]["parameters"]["height_cm"] == 45
        assert detector_configs[16]["parameters"]["height_cm"] == 55

        assert detector_configs[0]["parameters"]["dead_time_per_pulse"] == 5
        assert detector_configs[8]["parameters"]["dead_time_per_pulse"] == 10

        assert detector_configs[0]["parameters"]["latitude_deg"] == 35.0
        assert detector_configs[4]["parameters"]["latitude_deg"] == 45.0

        assert detector_configs[0]["parameters"]["longitude_deg"] == 253.4
        assert detector_configs[2]["parameters"]["longitude_deg"] == 263.4

        assert detector_configs[0]["parameters"]["elevation_m"] == 1620
        assert detector_configs[1]["parameters"]["elevation_m"] == 1720

    "************************************************************************"
    " Tests for Source Configs"
    "************************************************************************"

    def test_valid_validate_inject_config_source_single_string(self):
        # Modify base config for this test
        valid_config_config_source_single_string = copy.deepcopy(base_config)

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_string
        )

    def test_valid_validate_inject_config_multiple_source_single_string(self):
        # Modify base config for this test
        valid_config_config_multi_source_single_string = copy.deepcopy(base_config)
        valid_config_config_multi_source_single_string["sources"].append(
            {"isotope": "Ba133", "configurations": ["Ba133,100uC"]}
        )

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_multi_source_single_string
        )

    def test_valid_validate_inject_config_source_single_string_w_options(self):
        # Modify base config for this test
        valid_config_config_source_single_string_w_options = copy.deepcopy(base_config)
        valid_config_config_source_single_string_w_options["sources"][0][
            "configurations"
        ] = ["Am241,100.5uCi{ad=10,an=13}"]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_string_w_options
        )

    def test_valid_validate_inject_config_source_single_dict(self):
        # Modify base config for this test
        valid_config_config_source_single_dict = copy.deepcopy(base_config)
        valid_config_config_source_single_dict["sources"][0]["configurations"] = [
            {"name": "Am241", "activity": 37000.2, "activity_units": "Bq"}
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_dict
        )

    def test_valid_validate_inject_config_source_single_dict_w_shielding(self):
        # Modify base config for this test
        valid_config_config_source_single_dict_w_shielding = copy.deepcopy(base_config)
        valid_config_config_source_single_dict_w_shielding["sources"][0][
            "configurations"
        ] = [
            {
                "name": "Am241",
                "activity": 37000.2,
                "activity_units": "Bq",
                "shielding_atomic_number": 10.0,
                "shielding_aerial_density": 2.25,
            }
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_dict_w_shielding
        )

    def test_valid_validate_inject_config_source_single_dict_w_shielding_list(self):
        # Modify base config for this test
        valid_config_config_source_single_dict_w_shielding = copy.deepcopy(base_config)
        valid_config_config_source_single_dict_w_shielding["sources"][0][
            "configurations"
        ] = [
            {
                "name": "Am241",
                "activity": [37000.2, 37000.3],
                "activity_units": "Bq",
                "shielding_atomic_number": [20.0, 30, 40.0],
                "shielding_aerial_density": [2.25, 82, 26],
            }
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_dict_w_shielding
        )

    def test_valid_validate_inject_config_source_single_dict_w_shielding_sampling(self):
        # Modify base config for this test
        valid_config_config_source_single_dict_w_shielding = copy.deepcopy(base_config)
        valid_config_config_source_single_dict_w_shielding["sources"][0][
            "configurations"
        ] = [
            {
                "name": "Am241",
                "activity": 37000.2,
                "activity_units": "Bq",
                "shielding_atomic_number": {"mean": 10, "std": 1, "num_samples": 5},
                "shielding_aerial_density": {
                    "min": 2,
                    "max": 10,
                    "dist": "uniform",
                    "num_samples": 5,
                },
            }
        ]

        """Tests config validation."""
        assert self.return_validate_inject_config(
            valid_config_config_source_single_dict_w_shielding
        )

    def test_invalid_validate_inject_config_multiple_source_single_string(self):
        # Modify base config for this test
        valid_config_config_multi_source_single_string = copy.deepcopy(base_config)
        valid_config_config_multi_source_single_string["sources"][0] = {
            "not_an_isotope": "Am241",  # this is wrong
            "configurations": ["Am241,100.5uCi"],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )

        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": 1,  # this is wrong, has to be string
            "configurations": ["Am241,100.5uCi"],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )

        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "not_a_configurations": ["Am241,100.5uCi"],  # this is wrong
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )

        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": "Am241,100.5uCi",  # this is wrong, has to be array
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [20],  # this is wrong, can't be array of number
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [{"name": 20}],  # this is wrong, name has to be string
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [
                {
                    "name": "Am241",
                    "activity": "a string",  # this is wrong, can't be string
                }
            ],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [
                {"activity": "string"}  # this is wrong, name is required
            ],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [
                {
                    "name": "Am241",
                    "activity": 100.5,
                    "activity_units": "not_a_valid_unit",  # this is wrong, must be valid unts
                }
            ],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [
                {
                    "name": "Am241",
                    "activity": 100.5,
                    "activity_units": "uCi",
                    "shielding_atomic_number": "a string",  # this is wrong, can't be a string
                }
            ],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )
        valid_config_config_multi_source_single_string["sources"][0] = {
            "isotope": "Am241",
            "configurations": [
                {
                    "name": "Am241",
                    "activity": 100.5,
                    "activity_units": "uCi",
                    "shielding_atomic_number": 20,
                    "shielding_areal_density": "a string",  # this is wrong, can't be a string
                }
            ],
        }
        self.assertRaises(
            Exception,
            self.return_validate_inject_config,
            valid_config_config_multi_source_single_string,
        )

    # Source Expanded Configs
    def test_get_expanded_config_source_simple_string(self):
        # Modify base config for this test
        valid_config_source_with_simple_string = copy.deepcopy(base_config)

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi"]},
            ],
        }
        output = get_expanded_config(
            valid_config_source_with_simple_string,
        )

        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_source_complex_string(self):
        # Modify base config for this test
        valid_config_source_with_complex_string = copy.deepcopy(base_config)
        valid_config_source_with_complex_string["sources"][0]["configurations"] = [
            "Am241,100.5uCi{ad=10,an=13}"
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,100.5uCi{ad=10,an=13}"]},
            ],
        }
        output = get_expanded_config(valid_config_source_with_complex_string)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_source_name_activity_units(self):
        # Modify base config for this test
        valid_config_source_with_name_activity_units = copy.deepcopy(base_config)
        valid_config_source_with_name_activity_units["sources"][0]["configurations"] = [
            {"name": "Am241", "activity": 37000.2, "activity_units": "Bq"}
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {"isotope": "Am241", "configurations": ["Am241,37000.2Bq"]},
            ],
        }
        output = get_expanded_config(valid_config_source_with_name_activity_units)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_source_an_ad_as_numbers(self):
        # Modify base config for this test
        valid_config_source_with_an_ad_as_numbers = copy.deepcopy(base_config)
        valid_config_source_with_an_ad_as_numbers["sources"][0]["configurations"] = [
            {
                "name": "Am241",
                "activity": 100.5,
                "activity_units": "uCi",
                "shielding_atomic_number": 10.0,
                "shielding_aerial_density": 2.25,
            }
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {
                    "isotope": "Am241",
                    "configurations": ["Am241,100.5uCi{ad=2.25,an=10.0}"],
                },
            ],
        }
        output = get_expanded_config(valid_config_source_with_an_ad_as_numbers)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_source_ad_as_list(self):
        # Modify base config for this test
        valid_config_source_with_an_ad_as_numbers = copy.deepcopy(base_config)
        valid_config_source_with_an_ad_as_numbers["sources"][0]["configurations"] = [
            {
                "name": "Am241",
                "activity": 100.5,
                "activity_units": "uCi",
                "shielding_atomic_number": 20,
                "shielding_aerial_density": [2.25, 82.0, 26.0],
            }
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 42,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {
                    "isotope": "Am241",
                    "configurations": [
                        "Am241,100.5uCi{ad=2.25,an=20.0}",
                        "Am241,100.5uCi{ad=82.0,an=20.0}",
                        "Am241,100.5uCi{ad=26.0,an=20.0}",
                    ],
                },
            ],
        }
        output = get_expanded_config(valid_config_source_with_an_ad_as_numbers)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    def test_get_expanded_config_source_ad_as_samples(self):
        # Modify base config for this test
        valid_config_source_with_an_ad_as_samples = copy.deepcopy(base_config)
        valid_config_source_with_an_ad_as_samples["random_seed"] = 1
        valid_config_source_with_an_ad_as_samples["sources"][0]["configurations"] = [
            {
                "name": "Am241",
                "activity": 100.5,
                "activity_units": "uCi",
                "shielding_atomic_number": {"mean": 10, "std": 1, "num_samples": 5},
                "shielding_aerial_density": {
                    "min": 2,
                    "max": 10,
                    "dist": "uniform",
                    "num_samples": 5,
                },
            }
        ]

        # What we expect from get_expanded_config
        expected_expanded_config = {
            "random_seed": 1,
            "gamma_detector": {
                "name": "Generic\\NaI\\2x4x16",
                "parameters": {
                    "distance_cm": [1000],
                    "height_cm": [45],
                    "dead_time_per_pulse": [5],
                    "latitude_deg": [35.0],
                    "longitude_deg": [253.4],
                    "elevation_m": [1620],
                },
            },
            "sources": [
                {
                    "isotope": "Am241",
                    "configurations": [
                        "Am241,100.5uCi{ad=5.386611591780605,an=10.345584192064786}",
                        "Am241,100.5uCi{ad=8.621620750563533,an=10.345584192064786}",
                        "Am241,100.5uCi{ad=5.27359309095329,an=10.345584192064786}",
                        "Am241,100.5uCi{ad=6.396749501384476,an=10.345584192064786}",
                        "Am241,100.5uCi{ad=2.220472905944547,an=10.345584192064786}",
                        "Am241,100.5uCi{ad=5.386611591780605,an=10.821618143501158}",
                        "Am241,100.5uCi{ad=8.621620750563533,an=10.821618143501158}",
                        "Am241,100.5uCi{ad=5.27359309095329,an=10.821618143501158}",
                        "Am241,100.5uCi{ad=6.396749501384476,an=10.821618143501158}",
                        "Am241,100.5uCi{ad=2.220472905944547,an=10.821618143501158}",
                        "Am241,100.5uCi{ad=5.386611591780605,an=10.330437076183387}",
                        "Am241,100.5uCi{ad=8.621620750563533,an=10.330437076183387}",
                        "Am241,100.5uCi{ad=5.27359309095329,an=10.330437076183387}",
                        "Am241,100.5uCi{ad=6.396749501384476,an=10.330437076183387}",
                        "Am241,100.5uCi{ad=2.220472905944547,an=10.330437076183387}",
                        "Am241,100.5uCi{ad=5.386611591780605,an=8.696842768395639}",
                        "Am241,100.5uCi{ad=8.621620750563533,an=8.696842768395639}",
                        "Am241,100.5uCi{ad=5.27359309095329,an=8.696842768395639}",
                        "Am241,100.5uCi{ad=6.396749501384476,an=8.696842768395639}",
                        "Am241,100.5uCi{ad=2.220472905944547,an=8.696842768395639}",
                        "Am241,100.5uCi{ad=5.386611591780605,an=10.905355866673117}",
                        "Am241,100.5uCi{ad=8.621620750563533,an=10.905355866673117}",
                        "Am241,100.5uCi{ad=5.27359309095329,an=10.905355866673117}",
                        "Am241,100.5uCi{ad=6.396749501384476,an=10.905355866673117}",
                        "Am241,100.5uCi{ad=2.220472905944547,an=10.905355866673117}",
                    ],
                },
            ],
        }
        output = get_expanded_config(valid_config_source_with_an_ad_as_samples)
        self.assertEqual(
            output, expected_expanded_config, f"{output}!={expected_expanded_config}"
        )

    # def test_get_sources_setups_list_all(self):
    #     # Modify base config for this test
    #     valid_config_sources = copy.deepcopy(base_config)
    #     valid_config_sources["randsom_seed"] = 1
    #     valid_config_sources["sources"][0]["configurations"] = [
    #         {
    #             "name": "Am241",
    #             "activity": 100.5,
    #             "activity_units": "uCi",
    #             "shielding_atomic_number": 20,
    #             "shielding_aerial_density": [2.25, 82.0, 26.0],
    #         }
    #     ]
    #     valid_config_sources["sources"].append(
    #         {
    #             "isotope": "Ba133",
    #             "configurations": [
    #                 {
    #                     "name": "Ba133",
    #                     "activity": 200.5,
    #                     "activity_units": "Bq",
    #                     "shielding_atomic_number": [10, 20, 30],
    #                     "shielding_aerial_density": 3.35,
    #                 }
    #             ],
    #         }
    #     )
    #     expanded_config = get_expanded_config(valid_config_sources)
    #     sources_configs = get_sources_setups(expanded_config)

    #     # Asserts
    #     assert sources_configs["Am241"][0] == "Am241,100.5uCi{ad=2.25,an=20.0}"
    #     assert sources_configs["Am241"][1] == "Am241,100.5uCi{ad=82.0,an=20.0}"
    #     assert sources_configs["Am241"][2] == "Am241,100.5uCi{ad=26.0,an=20.0}"

    #     assert sources_configs["Ba133"][0] == "Ba133,200.5Bq{ad=3.35,an=10.0}"
    #     assert sources_configs["Ba133"][1] == "Ba133,200.5Bq{ad=3.35,an=20.0}"
    #     assert sources_configs["Ba133"][2] == "Ba133,200.5Bq{ad=3.35,an=30.0}"

    def test_get_inject_setups(self):
        # Modify base config for this test
        valid_get_inject_setups_list_all = copy.deepcopy(base_config)
        valid_get_inject_setups_list_all["random_seed"] = 1
        valid_get_inject_setups_list_all["gamma_detector"]["parameters"][
            "distance_cm"
        ] = [
            1000,
            2000,
        ]
        valid_get_inject_setups_list_all["gamma_detector"]["parameters"][
            "height_cm"
        ] = [
            45,
            55,
        ]
        valid_get_inject_setups_list_all["sources"][0]["configurations"] = [
            {
                "name": "Am241",
                "activity": 100.5,
                "activity_units": "uCi",
                "shielding_atomic_number": 20,
                "shielding_aerial_density": [2.25, 82.0, 26.0],
            }
        ]
        valid_get_inject_setups_list_all["sources"].append(
            {"isotope": "Ba133", "configurations": ["Ba133,100uC"]}
        )

        inject_setups = get_inject_setups(valid_get_inject_setups_list_all)

        # Asserts
        assert len(inject_setups) == 4
        assert inject_setups[0]["gamma_detector"]["parameters"]["distance_cm"] == 1000
        assert inject_setups[0]["gamma_detector"]["parameters"]["height_cm"] == 45
        assert inject_setups[0]["sources"][0]["isotope"] == "Am241"
        assert (
            inject_setups[0]["sources"][0]["configurations"][0]
            == "Am241,100.5uCi{ad=2.25,an=20.0}"
        )
        assert (
            inject_setups[0]["sources"][0]["configurations"][1]
            == "Am241,100.5uCi{ad=82.0,an=20.0}"
        )
        assert (
            inject_setups[0]["sources"][0]["configurations"][2]
            == "Am241,100.5uCi{ad=26.0,an=20.0}"
        )
        assert inject_setups[0]["sources"][1]["isotope"] == "Ba133"
        assert inject_setups[0]["sources"][1]["configurations"][0] == "Ba133,100uC"

        assert inject_setups[1]["gamma_detector"]["parameters"]["distance_cm"] == 1000
        assert inject_setups[1]["gamma_detector"]["parameters"]["height_cm"] == 55
        assert inject_setups[1]["sources"][0]["isotope"] == "Am241"
        assert (
            inject_setups[1]["sources"][0]["configurations"][0]
            == "Am241,100.5uCi{ad=2.25,an=20.0}"
        )
        assert (
            inject_setups[1]["sources"][0]["configurations"][1]
            == "Am241,100.5uCi{ad=82.0,an=20.0}"
        )
        assert (
            inject_setups[1]["sources"][0]["configurations"][2]
            == "Am241,100.5uCi{ad=26.0,an=20.0}"
        )
        assert inject_setups[1]["sources"][1]["isotope"] == "Ba133"
        assert inject_setups[1]["sources"][1]["configurations"][0] == "Ba133,100uC"

        assert inject_setups[2]["gamma_detector"]["parameters"]["distance_cm"] == 2000
        assert inject_setups[2]["gamma_detector"]["parameters"]["height_cm"] == 45
        assert inject_setups[2]["sources"][0]["isotope"] == "Am241"
        assert (
            inject_setups[2]["sources"][0]["configurations"][0]
            == "Am241,100.5uCi{ad=2.25,an=20.0}"
        )
        assert (
            inject_setups[2]["sources"][0]["configurations"][1]
            == "Am241,100.5uCi{ad=82.0,an=20.0}"
        )
        assert (
            inject_setups[2]["sources"][0]["configurations"][2]
            == "Am241,100.5uCi{ad=26.0,an=20.0}"
        )
        assert inject_setups[2]["sources"][1]["isotope"] == "Ba133"
        assert inject_setups[2]["sources"][1]["configurations"][0] == "Ba133,100uC"

        assert inject_setups[3]["gamma_detector"]["parameters"]["distance_cm"] == 2000
        assert inject_setups[3]["gamma_detector"]["parameters"]["height_cm"] == 55
        assert inject_setups[3]["sources"][0]["isotope"] == "Am241"
        assert (
            inject_setups[3]["sources"][0]["configurations"][0]
            == "Am241,100.5uCi{ad=2.25,an=20.0}"
        )
        assert (
            inject_setups[3]["sources"][0]["configurations"][1]
            == "Am241,100.5uCi{ad=82.0,an=20.0}"
        )
        assert (
            inject_setups[3]["sources"][0]["configurations"][2]
            == "Am241,100.5uCi{ad=26.0,an=20.0}"
        )
        assert inject_setups[3]["sources"][1]["isotope"] == "Ba133"
        assert inject_setups[3]["sources"][1]["configurations"][0] == "Ba133,100uC"
