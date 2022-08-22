# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the data module."""
import os
import tempfile
import unittest

from riid.data.labeling import BACKGROUND_LABEL, label_to_index_element
from riid.data.sampleset import SampleSet, _write_hdf, read_smpl
from riid.data.synthetic.static import get_dummy_sampleset


class TestData(unittest.TestCase):
    """Test class for Data module."""

    def setUp(self):
        """Test setup."""
        pass

    def _get_temp_smpl_path(self):
        default_temp_dir = tempfile._get_default_tempdir()
        temp_file_name = next(tempfile._get_candidate_names())
        return os.path.join(default_temp_dir, temp_file_name + ".smpl")

    def test_to_smpl_and_read_smpl(self):
        """Tests loading of SampleSets saved in hdf format."""
        file_path = self._get_temp_smpl_path()
        sampleset = get_dummy_sampleset()
        _write_hdf(sampleset, file_path)
        sampleset_out = read_smpl(file_path)
        compare_samplesets(self, sampleset, sampleset_out)
        os.remove(file_path)

    def test_label_to_index_element(self):
        RAD_SOURCES = {
            BACKGROUND_LABEL: (BACKGROUND_LABEL, BACKGROUND_LABEL),
            "K40,100uC": ("K40", "NORM"),
            "K40inCargo1": ("K40", "NORM"),
            "K40inCargo2": ("K40", "NORM"),
            "K40inCargo3": ("K40", "NORM"),
            "K40inCargo4": ("K40", "NORM"),
            "Ra226,100uC": ("Ra226", "NORM"),
            "Ra226inCargo1": ("Ra226", "NORM"),
            "Ra226inCargo2": ("Ra226", "NORM"),
            "Ra226inCargo3": ("Ra226", "NORM"),
            "Ra226inSilica1": ("Ra226", "NORM"),
            "Ra226inSilica2": ("Ra226", "NORM"),
            "Ra226,100uC {60,100}": ("Ra226", "NORM"),
            "Th232,100uC": ("Th232", "NORM"),
            "Th232inCargo1": ("Th232", "NORM"),
            "Th232inCargo2": ("Th232", "NORM"),
            "Th232inCargo3": ("Th232", "NORM"),
            "Th232inSilica1": ("Th232", "NORM"),
            "Th232inSilica2": ("Th232", "NORM"),
            "Th232,100uC {60,100}": ("Th232", "NORM"),
            "ThPlate": ("Th232", "NORM"),
            "ThPlate+Thxray,10uC": ("Th232", "NORM"),
            "fiestaware": ("U238", "SNM"),
            "U232,100uC": ("U232", "SNM"),
            "U232,100uC {10,10}": ("U232", "SNM"),
            "U232,100uC {10,30}": ("U232", "SNM"),
            "U232,100uC {13,50}": ("U232", "SNM"),
            "U232,100uC {26,30}": ("U232", "SNM"),
            "U232,100uC {26,60}": ("U232", "SNM"),
            "U232inLeadAndPine": ("U232", "SNM"),
            "U232inLeadAndFe": ("U232", "SNM"),
            "U232inPineAndFe": ("U232", "SNM"),
            "U232,100uC {82,30}": ("U232", "SNM"),
            "1kgU233At1yr": ("U233", "SNM"),
            "1kgU233InFeAt1yr": ("U233", "SNM"),
            "1kgU233At50yr": ("U233", "SNM"),
            "1kgU233InFeAt50yr": ("U233", "SNM"),
            "1kgU235": ("U235", "SNM"),
            "1kgU235inFe": ("U235", "SNM"),
            "1kgU235inPine": ("U235", "SNM"),
            "1kgU235inPineAndFe": ("U235", "SNM"),
            "1gNp237,1kC": ("Np237", "SNM"),
            "1kgNp237": ("Np237", "SNM"),
            "Np237inFe1": ("Np237", "SNM"),
            "Np237inFe4": ("Np237", "SNM"),
            "Np237Shielded2": ("Np237", "SNM"),
            "1KGU238": ("U238", "SNM"),
            "1KGU238Clad": ("U238", "SNM"),
            "1KGU238inPine": ("U238", "SNM"),
            "1KGU238inPine2": ("U238", "SNM"),
            "1KGU238inPine3": ("U238", "SNM"),
            "1KGU238inFe": ("U238", "SNM"),
            "1KGU238inPineAndFe": ("U238", "SNM"),
            "1KGU238inW": ("U238", "SNM"),
            "U238Fiesta": ("U238", "SNM"),
            "Uxray,100uC": ("U238", "SNM"),
            "DUOxide": ("U238", "SNM"),
            "1gPu239": ("Pu239", "SNM"),
            "1kgPu239": ("Pu239", "SNM"),
            "1kgPu239,1C {40,4}": ("Pu239", "SNM"),
            "1kgPu239inFe": ("Pu239", "SNM"),
            "1kgPu239inPine": ("Pu239", "SNM"),
            "1kgPu239InFeAndPine": ("Pu239", "SNM"),
            "1kgPu239inW": ("Pu239", "SNM"),
            "Pu238,100uC": ("Pu238", "SNM"),
            "Pu238,100uC {10,5}": ("Pu238", "SNM"),
            "Pu238,100uC {26,10}": ("Pu238", "SNM"),
            "Am241,100uC": ("Am241", "Industrial"),
            "Am241,100UC {13,10}": ("Am241", "Industrial"),
            "Am241,100UC {13,30}": ("Am241", "Industrial"),
            "Am241,100UC {26,5}": ("Am241", "Industrial"),
            "Am241,100UC {26,20}": ("Am241", "Industrial"),
            "Am241,100UC {26,50}": ("Am241", "Industrial"),
            "Am241,100UC {50,2}": ("Am241", "Industrial"),
            "Na22,100uC {10,10}": ("Na22", "Industrial"),
            "Na22,100uC": ("Na22", "Industrial"),
            "Na22,100uC {10,30}": ("Na22", "Industrial"),
            "Na22,100uC {74,20}": ("Na22", "Industrial"),
            "Co57,100uC {13,10}": ("Co57", "Industrial"),
            "Co57,100uC": ("Co57", "Industrial"),
            "Co60,100uC": ("Co60", "Industrial"),
            "Co60,100uC {10,10}": ("Co60", "Industrial"),
            "Co60,100uC {10,30}": ("Co60", "Industrial"),
            "Co60,100uC {26,20}": ("Co60", "Industrial"),
            "Co60,100uC {26,40}": ("Co60", "Industrial"),
            "Co60,100uC {82,30}": ("Co60", "Industrial"),
            "Co60,100uC {82,60}": ("Co60", "Industrial"),
            "Y88,100uC": ("Y88", "Industrial"),
            "Y88,100uC {10,50}": ("Y88", "Industrial"),
            "Y88,100uC {26,30}": ("Y88", "Industrial"),
            "Y88,100uC {80,50}": ("Y88", "Industrial"),
            "Ba133,100uC": ("Ba133", "Industrial"),
            "Ba133,100uC {10,20}": ("Ba133", "Industrial"),
            "Ba133,100uC {26,10}": ("Ba133", "Industrial"),
            "Ba133,100uC {26,30}": ("Ba133", "Industrial"),
            "Ba133,100uC {74,20}": ("Ba133", "Industrial"),
            "Ba133,100uC {50,30}": ("Ba133", "Industrial"),
            "Cs137,100uC": ("Cs137", "Industrial"),
            "Cs137,100uC {6,2}": ("Cs137", "Industrial"),
            "Cs137,100uC {26,10}": ("Cs137", "Industrial"),
            "Cs137,100uC {82,10}": ("Cs137", "Industrial"),
            "Cs137,100uC {13,240;26,,}": ("Cs137", "Industrial"),
            "Cs137,100uC {13,240;26,1}": ("Cs137", "Industrial"),
            "Cs137InPine": ("Cs137", "Industrial"),
            "Cs137InLead": ("Cs137", "Industrial"),
            "Cs137InGradedShield1": ("Cs137", "Industrial"),
            "Cs137InGradedShield2": ("Cs137", "Industrial"),
            "Eu152,100uC": ("Eu152", "Industrial"),
            "Eu152,100uC {10,10}": ("Eu152", "Industrial"),
            "Eu152,100uC {10,30}": ("Eu152", "Industrial"),
            "Eu152,100uC {30,50}": ("Eu152", "Industrial"),
            "Eu152,100uC {74,20}": ("Eu152", "Industrial"),
            "Eu154,100uC": ("Eu154", "Industrial"),
            "Eu154,100uC {10,10}": ("Eu154", "Industrial"),
            "Eu154,100uC {74,20}": ("Eu154", "Industrial"),
            "Ho166m,100uC": ("Ho166m", "Industrial"),
            "Ho166m,100uC {10,20}": ("Ho166m", "Industrial"),
            "Ho166m,100uC {26,20}": ("Ho166m", "Industrial"),
            "Ho166m,100uC {74,30}": ("Ho166m", "Industrial"),
            "Ir192,100uC": ("Ir192", "Industrial"),
            "Ir192,100uC {10,20}": ("Ir192", "Industrial"),
            "Ir192,100uC {26,40}": ("Ir192", "Industrial"),
            "Ir192,100uC {26,100}": ("Ir192", "Industrial"),
            "Ir192,100uC {82,30}": ("Ir192", "Industrial"),
            "Ir192,100uC {82,160}": ("Ir192", "Industrial"),
            "Ir192Shielded1": ("Ir192", "Industrial"),
            "Ir192Shielded2": ("Ir192", "Industrial"),
            "Ir192Shielded3": ("Ir192", "Industrial"),
            "Bi207,100uC": ("Bi207", "Industrial"),
            "Bi207,100uC {10,30}": ("Bi207", "Industrial"),
            "Bi207,100uC {26,10}": ("Bi207", "Industrial"),
            "Cf249,100uC": ("Cf249", "Industrial"),
            "Cs137,1mC": ("Cs137", "Industrial"),
            "F18,100uC": ("F18", "Medical"),
            "F18,100uC {10,20}": ("F18", "Medical"),
            "F18,100uC {10,50}": ("F18", "Medical"),
            "F18,100uC {26,30}": ("F18", "Medical"),
            "Ga67,100UC": ("Ga67", "Medical"),
            "Ga67,100UC {6,10}": ("Ga67", "Medical"),
            "Ga67,100UC {6,20}": ("Ga67", "Medical"),
            "Ga67,100UC {10,30}": ("Ga67", "Medical"),
            "Ga67,100UC {50,16}": ("Ga67", "Medical"),
            "Ga67,100UC {82,20}": ("Ga67", "Medical"),
            "Mo99,100uC": ("Mo99", "Medical"),
            "Mo99,100uC {26,20}": ("Mo99", "Medical"),
            "Mo99,100uC {50,40}": ("Mo99", "Medical"),
            "Tc99m,100uC": ("Tc99m", "Medical"),
            "Tc99m,100uC {7,10}": ("Tc99m", "Medical"),
            "Tc99m,100uC {10,20}": ("Tc99m", "Medical"),
            "Tc99m,100uC {13,30}": ("Tc99m", "Medical"),
            "Tc99m,100uC {26,30}": ("Tc99m", "Medical"),
            "In111,100uC": ("In111", "Medical"),
            "In111,100uC {10,20}": ("In111", "Medical"),
            "In111,100uC {50,20}": ("In111", "Medical"),
            "I123,100uC": ("I123", "Medical"),
            "I123,100uC {10,30}": ("I123", "Medical"),
            "I125,100uC": ("I125", "Medical"),
            "I131,100uC": ("I131", "Medical"),
            "I131,100uC {10,10}": ("I131", "Medical"),
            "I131,100uC {10,30}": ("I131", "Medical"),
            "I131,100uC {16,50}": ("I131", "Medical"),
            "I131,100uC {20,20}": ("I131", "Medical"),
            "I131,100uC {82,10}": ("I131", "Medical"),
            "I131,100uC {10,20;50,5}": ("I131", "Medical"),
            "Tl201,100uC": ("Tl201", "Medical"),
            "Tl201,100uC {8,40}": ("Tl201", "Medical"),
            "Tl201,100uC {10,10}": ("Tl201", "Medical"),
            "Tl201,100uC {10,30}": ("Tl201", "Medical"),
            "Tl201,100uC {26,10}": ("Tl201", "Medical"),
            "Sr90InPoly1,1C": ("Sr90", "Industrial"),
            "Sr90InPoly10,1C": ("Sr90", "Industrial"),
            "Sr90InFe,1C": ("Sr90", "Industrial"),
            "Sr90InSn,1C": ("Sr90", "Industrial"),
            "pu239_1yr": ("Pu239", "SNM"),
            "modified_berpball": ("Pu239", "SNM"),
            "pu239_5yr": ("Pu239", "SNM"),
            "pu239_10yr": ("Pu239", "SNM"),
            "pu239_25yr": ("Pu239", "SNM"),
            "pu239_50yr": ("Pu239", "SNM"),
            "1gPuWG_0.5yr,3{an=10,ad=5}": ("Pu239", "SNM"),
            "1kg HEU + 800uCi Cs137": ("U235 + U238 + Cs137", "SNM"),
            "WGPu + Cs137": ("Pu239 + Cs137", "SNM"),
            "10 yr WGPu in Fe": ("Pu239", "SNM")
        }
        for seed, (isotope, category) in RAD_SOURCES.items():
            actual_category, actual_isotope, _ = label_to_index_element(
                seed,
                label_level="Seed"
            )
            msg = f"{isotope} ({category}) != {actual_isotope} ({actual_category})"
            self.assertEqual(isotope, actual_isotope, msg)
            self.assertEqual(category, actual_category, msg)


def compare_samplesets(unit_test, ss1: SampleSet, ss2: SampleSet):
    """ Compares two SampleSets.

        Extra data, info, predictions, and even sources can all be lost
        in the following conversion: SMPL -> PCF -> SMPL.
        However, no information should be lost in the following
        conversion: PCF -> SMPL -> PCF
    """
    unit_test.assertEqual(
        ss1.measured_or_synthetic,
        ss2.measured_or_synthetic,
        "Measured or synthetic not equal"
    )
    unit_test.assertEqual(
        ss1.detector_hash,
        ss2.detector_hash,
        "Detector hash not equal"
    )
    unit_test.assertEqual(
        ss1.neutron_detector_hash,
        ss2.neutron_detector_hash,
        "Neutron detector hash not equal"
    )
    unit_test.assertEqual(
        ss1.detector_info,
        ss2.detector_info,
        "Detector info not equal"
    )
    unit_test.assertEqual(
        ss1.synthesis_info,
        ss2.synthesis_info,
        "Synthesis info not equal"
    )

    unit_test.assertTrue(
        ss1.spectra.equals(ss2.spectra),
        "Spectra are not equal"
    )
    unit_test.assertTrue(
        ss1.sources.equals(ss2.sources),
        "Sources are not equal"
    )
    unit_test.assertTrue(
        ss1.info.equals(ss2.info),
        "Sample info is not equal"
    )
    unit_test.assertTrue(
        ss1.prediction_probas.equals(ss2.prediction_probas),
        "Predictions are not equal"
    )
    unit_test.assertTrue(
        ss1.extra_data.equals(ss2.extra_data),
        "Extra data is not equal"
    )


if __name__ == '__main__':
    unittest.main()
