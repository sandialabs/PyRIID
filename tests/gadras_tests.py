# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module tests the gadras module."""
import pathlib
import unittest

from riid.gadras import pcf_to_smpl, smpl_to_pcf


class TestGadras(unittest.TestCase):
    """Test class for Gadras.
    """
    def setUp(self):
        """Test setup.
        """
        self.GADRAS_DB_PCF_FILE_NAME = "_gadras_db.pcf"
        self.GADRAS_DB_PCF_FILE_PATH = pathlib.PurePath.joinpath(
            pathlib.Path(__file__).parent.absolute(),
            self.GADRAS_DB_PCF_FILE_NAME
        )
        with open(self.GADRAS_DB_PCF_FILE_PATH, "rb") as fin:
            self.original_pcf_contents = fin.read()

        self.recreated_pcf_path = "recreated_pcf.pcf"

    def tearDown(self):
        """Clears out any created files.
        """
        pathlib.Path(self.recreated_pcf_path).unlink()

    def test_bytes_to_smpl_to_pcf_to_bytes(self):
        """Tests PCF-to-SampleSet then SampleSet-to-PCF.
        """
        # Load original pcf contents to SampleSet
        ss = pcf_to_smpl(self.GADRAS_DB_PCF_FILE_PATH, True)
        # Save SampleSet out to .pcf file
        smpl_to_pcf(ss, self.recreated_pcf_path)

        # Load bytes of original and recreated files
        contents = []
        for path in [self.GADRAS_DB_PCF_FILE_PATH, self.recreated_pcf_path]:
            with open(path, "rb") as fin:
                contents.append(fin.read())

        self.assertTrue(
            contents[0] == contents[1],
            "Expect pcf data to be unchanged by reading and writing"
        )


if __name__ == '__main__':
    unittest.main()
