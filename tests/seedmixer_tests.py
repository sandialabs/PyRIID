# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import unittest

import numpy as np

from riid.data.labeling import BACKGROUND_LABEL
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import get_dummy_sampleset


class TestSeedMixer(unittest.TestCase):
    """Test seed mixing functionality of SampleSet.
    """
    def setUp(self):
        self.ss = get_dummy_sampleset(as_seeds=True)
        self.ss.normalize()
        self.sources = self.ss.get_labels().values

        self.mixer2 = SeedMixer(
            mixture_size=2,
            min_source_contribution=0.2,
        )
        self.two_mix_seeds_ss = self.mixer2.generate(self.ss, n_samples=20)

        self.mixer3 = SeedMixer(
            mixture_size=3,
            min_source_contribution=0.1,
        )
        self.three_mix_seeds_ss = self.mixer3.generate(self.ss, n_samples=20)

    def test_mixture_combinations(self):
        # check that each mixture contains unique isotopes and the correct mixture size
        for each in self.two_mix_seeds_ss.get_source_contributions(target_level="Seed").values:
            mix_sources = self.sources[np.nonzero(each)]
            self.assertEqual(np.unique(mix_sources, return_counts=True)[1].max(), 1)
            self.assertEqual(np.count_nonzero(each), 2)

        for each in self.three_mix_seeds_ss.get_source_contributions(target_level="Seed").values:
            mix_sources = self.sources[np.nonzero(each)]
            self.assertEqual(np.unique(mix_sources, return_counts=True)[1].max(), 1)
            self.assertEqual(np.count_nonzero(each), 3)

    def test_mixture_ratios(self):
        # cehck for valid probability distribution and minimum contribution amount
        for each in self.two_mix_seeds_ss.get_source_contributions(target_level="Isotope").values:
            self.assertAlmostEqual(each.sum(), 1.0)
            self.assertTrue(each[np.nonzero(each)].min() >= self.mixer2.min_source_contribution)

        for each in self.three_mix_seeds_ss.get_source_contributions(target_level="Isotope").values:
            self.assertAlmostEqual(each.sum(), 1.0)
            self.assertTrue(each[np.nonzero(each)].min() >= self.mixer3.min_source_contribution)

    def test_mixture_number(self):
        # check that number of samples is less than largest possible combinations
        # (worst case scenario)
        self.assertEqual(self.two_mix_seeds_ss.n_samples, 20)
        self.assertEqual(self.two_mix_seeds_ss.n_samples, 20)

    def test_mixture_pdf(self):
        # check that each mixture sums to one
        for sample in range(self.two_mix_seeds_ss.n_samples):
            self.assertAlmostEqual(self.two_mix_seeds_ss.spectra.values[sample, :].sum(), 1.0)

        for sample in range(self.three_mix_seeds_ss.n_samples):
            self.assertAlmostEqual(self.three_mix_seeds_ss.spectra.values[sample, :].sum(), 1.0)

    def test_mixture_columns(self):
        # check that all mixtures are created as expected
        mixer = SeedMixer(
            mixture_size=3,
            min_source_contribution=0.1
        )
        mix_ss = mixer.generate(seeds_ss=self.ss, n_samples=100)

        fg_ss = self.ss[self.ss.get_labels() != BACKGROUND_LABEL]
        fg_ss.sources.drop(
            BACKGROUND_LABEL,
            axis=1,
            level="Isotope",
            inplace=True,
            errors='ignore'
        )
        spectra_dict = fg_ss.spectra.values.T
        recon_spectra = np.dot(spectra_dict, mix_ss.sources.values.T).T

        test_recon_spectra = np.zeros_like(recon_spectra)
        for row, each in enumerate(mix_ss.sources.values):
            source_inds = np.nonzero(each)[0]
            source_ratios = each[source_inds]
            for idx, ratio in enumerate(source_ratios):
                test_recon_spectra[row, :] += self.ss.spectra.loc[source_inds[idx], :]*ratio

        print('testing')
        self.assertTrue(np.allclose(recon_spectra, test_recon_spectra))


if __name__ == '__main__':
    unittest.main()
