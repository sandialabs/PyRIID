# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import unittest

import numpy as np

from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import get_dummy_seeds


class TestSeedMixer(unittest.TestCase):
    """Test seed mixing functionality of SampleSet.
    """
    def setUp(self):
        np.random.seed(42)

        self.ss, _ = get_dummy_seeds().split_fg_and_bg()
        self.ss.normalize()
        self.sources = self.ss.get_labels().values

        self.two_mix_seeds_ss = SeedMixer(
            self.ss,
            mixture_size=2,
            dirichlet_alpha=10,
        ).generate(n_samples=20)

        self.three_mix_seeds_ss = SeedMixer(
            self.ss,
            mixture_size=3,
            dirichlet_alpha=10,
        ).generate(n_samples=20)

    def test_mixture_combinations(self):
        # check that each mixture contains unique isotopes and the correct mixture size
        two_mix_isotopes = [
            x.split(" + ")
            for x in self.two_mix_seeds_ss.get_labels(target_level="Isotope", max_only=False)
        ]
        self.assertTrue(all([len(set(x)) == 2 for x in two_mix_isotopes]))

        three_mix_isotopes = [
            x.split(" + ")
            for x in self.three_mix_seeds_ss.get_labels(target_level="Isotope", max_only=False)
        ]
        self.assertTrue(all([len(set(x)) == 3 for x in three_mix_isotopes]))

    def test_mixture_ratios(self):
        # check for valid probability distribution
        for each in self.two_mix_seeds_ss.get_source_contributions(target_level="Isotope").values:
            self.assertAlmostEqual(each.sum(), 1.0)

        for each in self.three_mix_seeds_ss.get_source_contributions(target_level="Isotope").values:
            self.assertAlmostEqual(each.sum(), 1.0)

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
        mix_ss = SeedMixer(self.ss, mixture_size=3).generate(n_samples=100)
        recon_spectra = np.dot(
            self.ss.spectra.values.T,
            mix_ss.sources.values.T
        ).T
        test_recon_spectra = np.zeros_like(recon_spectra)
        for row, each in enumerate(mix_ss.sources.values):
            source_inds = np.nonzero(each)[0]
            source_ratios = each[source_inds]
            for idx, ratio in enumerate(source_ratios):
                test_recon_spectra[row, :] += self.ss.spectra.loc[source_inds[idx], :] * ratio

        np.allclose(recon_spectra, test_recon_spectra)


if __name__ == '__main__':
    unittest.main()
