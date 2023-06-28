# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import unittest

import numpy as np
from scipy.spatial.distance import jensenshannon
from riid.data.sampleset import SampleSet

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer


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

    def test_spectrum_construction_3seeds_2mix(self):
        _, bg_seeds_ss = get_dummy_seeds(n_channels=16).split_fg_and_bg()
        mixed_bg_ss = SeedMixer(bg_seeds_ss, mixture_size=2).generate(100)
        spectral_distances = _get_spectral_distances(bg_seeds_ss, mixed_bg_ss)
        self.assertTrue(all(spectral_distances == 0))

    def test_spectrum_construction_3seeds_3mix(self):
        _, bg_seeds_ss = get_dummy_seeds(n_channels=16).split_fg_and_bg()
        mixed_bg_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(100)
        spectral_distances = _get_spectral_distances(bg_seeds_ss, mixed_bg_ss)
        self.assertTrue(all(spectral_distances == 0))

    def test_spectrum_construction_2seeds_2mix(self):
        _, bg_seeds_ss = get_dummy_seeds(n_channels=16).split_fg_and_bg(
            bg_seed_names=SampleSet.DEFAULT_BG_SEED_NAMES[1:3]
        )
        mixed_bg_ss = SeedMixer(bg_seeds_ss, mixture_size=2).generate(100)
        spectral_distances = _get_spectral_distances(bg_seeds_ss, mixed_bg_ss)
        self.assertTrue(all(spectral_distances == 0))

    def test_spectrum_construction_2seeds_2mix_error(self):
        _, bg_seeds_ss = get_dummy_seeds(n_channels=16).split_fg_and_bg(
            bg_seed_names=SampleSet.DEFAULT_BG_SEED_NAMES[1:3]
        )
        mixer = SeedMixer(bg_seeds_ss, mixture_size=3)
        self.assertRaises(ValueError, mixer.generate, 100)


def _get_spectral_distances(seeds_ss, mixed_ss):
    recon_spectra = np.dot(
        seeds_ss.spectra.values.T,
        mixed_ss.sources.values.T
    ).T
    spectral_distances = jensenshannon(
        recon_spectra.astype(np.float32),
        mixed_ss.spectra.values.astype(np.float32),
        axis=1
    )
    return spectral_distances


if __name__ == '__main__':
    unittest.main()
