# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the bayes module."""
import os
import unittest

import numpy as np
import pandas as pd

from riid.data.sampleset import SampleSet
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models import PyRIIDModel
from riid.models.bayes import (NegativeSpectrumError, PoissonBayesClassifier,
                               ZeroTotalCountsError)
from riid.models.neural_nets import (LabelProportionEstimator, MLPClassifier,
                                     MultiEventClassifier)
from riid.models.neural_nets.arad import ARADLatentPredictor, ARADv1, ARADv2


class TestModels(unittest.TestCase):
    """Test class for PyRIID models."""
    def setUp(self):
        """Test setup."""
        pass

    @classmethod
    def setUpClass(self):
        self.seeds_ss = get_dummy_seeds(n_channels=128)
        self.fg_seeds_ss, self.bg_seeds_ss = self.seeds_ss.split_fg_and_bg()
        self.mixed_bg_seeds_ss = SeedMixer(self.bg_seeds_ss, mixture_size=3).generate(1)
        self.static_synth = StaticSynthesizer(samples_per_seed=5)
        self.train_ss, _ = self.static_synth.generate(self.fg_seeds_ss, self.mixed_bg_seeds_ss,
                                                      verbose=False)
        self.train_ss.prediction_probas = self.train_ss.sources
        self.train_ss.normalize()
        self.test_ss, _ = self.static_synth.generate(self.fg_seeds_ss, self.mixed_bg_seeds_ss,
                                                     verbose=False)
        self.test_ss.normalize()

    @classmethod
    def tearDownClass(self):
        pass

    def test_pb_constructor_errors(self):
        """Testing for constructor errors when different arguments are provided."""
        pb_model = PoissonBayesClassifier()

        # Empty argument provided
        spectra = np.array([])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ValueError, pb_model.fit, ss)

        # Negative channel argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [1, 1, -1, 1],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(NegativeSpectrumError, pb_model.fit, ss)

        # Zero total counts argument provided
        spectra = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        ss = SampleSet()
        ss.spectra = pd.DataFrame(spectra)
        self.assertRaises(ZeroTotalCountsError, pb_model.fit, ss)

    def test_pb_predict(self):
        """Tests the constructor with a valid SampleSet."""
        seeds_ss = get_dummy_seeds()
        fg_seeds_ss, bg_seeds_ss = seeds_ss.split_fg_and_bg()
        bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(1)

        # Create the PoissonBayesClassifier
        pb_model = PoissonBayesClassifier()
        pb_model.fit(fg_seeds_ss)

        # Get test samples
        gss = StaticSynthesizer(
            samples_per_seed=1,
            live_time_function_args=(4, 4),
            snr_function_args=(10, 10),
            rng=np.random.default_rng(42),
            return_fg=True,
            return_gross=True,
        )
        test_fg_ss, test_gross_ss = gss.generate(fg_seeds_ss, bg_seeds_ss, verbose=False)
        test_bg_ss = test_gross_ss - test_fg_ss

        # Predict
        pb_model.predict(test_gross_ss, test_bg_ss)

        truth_labels = fg_seeds_ss.get_labels()
        predicted_labels = test_gross_ss.get_predictions()
        assert (truth_labels == predicted_labels).all()

    def test_pb_fit_save_load(self):
        _test_model_fit_save_load_predict(self, PoissonBayesClassifier, None, self.fg_seeds_ss)

    def test_mlp_fit_save_load_predict(self):
        _test_model_fit_save_load_predict(self, MLPClassifier, self.test_ss, self.train_ss,
                                          epochs=1)

    def test_mec_fit_save_load_predict(self):
        test_copy_ss = self.test_ss[:]
        test_copy_ss.prediction_probas = test_copy_ss.sources
        _test_model_fit_save_load_predict(
            self,
            MultiEventClassifier,
            [test_copy_ss],
            [self.train_ss],
            self.train_ss.sources.groupby(axis=1, level="Isotope", sort=False).sum(),
            epochs=1
        )

    def test_lpe_fit_save_load_predict(self):
        _test_model_fit_save_load_predict(self, LabelProportionEstimator, self.test_ss,
                                          self.fg_seeds_ss, self.train_ss, epochs=1)

    def test_aradv1_fit_save_load_predict(self):
        _test_model_fit_save_load_predict(self, ARADv1, self.test_ss, self.train_ss, epochs=1)

    def test_aradv2_fit_save_load_predict(self):
        _test_model_fit_save_load_predict(self, ARADv2, self.test_ss, self.train_ss, epochs=1)

    def test_alp_fit_save_load_predict(self):
        arad_v2 = ARADv2()
        arad_v2.fit(self.train_ss, epochs=1)
        _test_model_fit_save_load_predict(self, ARADLatentPredictor, self.test_ss, arad_v2.model,
                                          self.train_ss, target_info_columns=["snr"], epochs=1)


def _try_remove_model_and_info(model_path: str):
    if os.path.exists(model_path):
        if os.path.isdir(model_path):
            os.rmdir(model_path)
        else:
            os.remove(model_path)


def _test_model_fit_save_load_predict(test_case: unittest.TestCase, model_class: PyRIIDModel,
                                      test_ss: SampleSet = None, *args_for_fit, **kwargs_for_fit):
    m1 = model_class()
    m2 = model_class()

    m1.fit(*args_for_fit, **kwargs_for_fit)

    model_path = m1._temp_file_path

    _try_remove_model_and_info(model_path)
    test_case.assertRaises(ValueError, m2.load, model_path)

    m1.save(model_path)
    test_case.assertRaises(ValueError, m1.save, model_path)

    m2.load(model_path)
    _try_remove_model_and_info(model_path)

    if test_ss is not None:
        m1.predict(test_ss)


if __name__ == "__main__":
    unittest.main()
