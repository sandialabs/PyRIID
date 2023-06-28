# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the gadras module."""
import unittest

import pandas as pd
from riid.data.synthetic import get_dummy_seeds
from riid.gadras.pcf import (_pack_compressed_text_buffer,
                             _unpack_compressed_text_buffer)


class TestGadras(unittest.TestCase):
    """Test class for Gadras."""

    def test_pcf_header_formatting(self):
        """Tests the PCF header information is properly packed.

        String formatting note:
            f"{'hello':60.50}"
                ^      ^  ^
                |      |  '> The number of letters to allow from the input value
                |      '> The length of the string
                '> The input value to format

        """
        FIELD_LENGTH = 10
        test_cases = [
            (
                "tttttttt  ",  "ddddddddd ",    "ssssssssss",
                "tttttttt",    "ddddddddd",     "ssssssssss",
                "tttttttt  ddddddddd ssssssssss",
            ),
            (
                "tttttttttt",   "dddddddddd",   "ssssssssss+",
                "tttttttttt",   "",             "ssssssssss+",
                "ÿttttttttttÿÿssssssssss+      "
            ),
            (
                "tttttttt  ",   "dddddddd  ",   "ssssssssss+",
                "tttttttt",     "dddddddd",     "ssssssssss+",
                "ÿttttttttÿddddddddÿssssssssss+"
            ),
            (
                "tt        ",   "dddddddddd+", "ssssssssss++",
                "tt",           "dddddddddd+", "ssssssssss++",
                "ÿttÿdddddddddd+ÿssssssssss++  "
            ),
        ]
        for case in test_cases:
            title, desc, source, \
                expected_title, expected_desc, expected_source, \
                expected_ctb = case
            actual_ctb = _pack_compressed_text_buffer(
                title,
                desc,
                source,
                field_len=FIELD_LENGTH
            )
            actual_title, actual_desc, actual_source = _unpack_compressed_text_buffer(
                actual_ctb,
                field_len=FIELD_LENGTH
            )
            self.assertEqual(expected_title, actual_title)
            self.assertEqual(expected_desc, actual_desc)
            self.assertEqual(expected_source, actual_source)
            self.assertEqual(expected_ctb, actual_ctb)

    def test_to_pcf_with_various_sources_dataframes(self):
        TEMP_PCF_PATH = "temp.pcf"

        # With all levels
        ss = get_dummy_seeds()
        ss.to_pcf(TEMP_PCF_PATH, verbose=False)

        # Without seed level (only category and isotope)
        ss = get_dummy_seeds()
        ss.sources.columns.droplevel("Seed")
        ss.to_pcf(TEMP_PCF_PATH, verbose=False)

        # Without seed and isotope levels (only category)
        ss = get_dummy_seeds()
        ss.sources.columns.droplevel("Seed")
        ss.sources.columns.droplevel("Isotope")
        ss.to_pcf(TEMP_PCF_PATH, verbose=False)

        # With no sources
        ss = get_dummy_seeds()
        ss.sources = pd.DataFrame()
        ss.to_pcf(TEMP_PCF_PATH, verbose=False)


if __name__ == '__main__':
    unittest.main()
