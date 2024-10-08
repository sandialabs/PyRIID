# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the sampleset module."""
import unittest

import numpy as np
import pandas as pd

from riid import SeedMixer, StaticSynthesizer, get_dummy_seeds
from riid.data import InvalidSeedError, get_expected_spectra
from riid.data.synthetic.base import (Synthesizer,
                                      get_merged_sources_samplewise,
                                      get_samples_per_seed)


class TestStaticSynthesis(unittest.TestCase):
    """Test class for static synthesis.
    """
    def test_random_state_output(self):
        fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
        mixed_bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(10)

        static_syn = StaticSynthesizer(
            samples_per_seed=2,
            snr_function="log10",
            rng=np.random.default_rng(1),
            return_fg=True,
            return_gross=True,
        )
        fg_1, gross_1 = static_syn.generate(
            fg_seeds_ss=fg_seeds_ss,
            bg_seeds_ss=mixed_bg_seeds_ss,
            verbose=False,
        )

        static_syn = StaticSynthesizer(
            samples_per_seed=2,
            snr_function="log10",
            return_fg=True,
            return_gross=True,
            rng=np.random.default_rng(1),
        )

        fg_2, gross_2 = static_syn.generate(
            fg_seeds_ss=fg_seeds_ss,
            bg_seeds_ss=mixed_bg_seeds_ss,
            verbose=False
        )

        static_syn = StaticSynthesizer(
            samples_per_seed=2,
            snr_function="log10",
            return_fg=True,
            return_gross=True,
            rng=np.random.default_rng(2),
        )

        fg_3, gross_3 = static_syn.generate(
            fg_seeds_ss=fg_seeds_ss,
            bg_seeds_ss=mixed_bg_seeds_ss,
            verbose=False
        )

        self.assertEqual(fg_1, fg_2)  # used the same random_state
        self.assertEqual(gross_1, gross_2)  # used the same random_state

        self.assertNotEqual(fg_3, fg_2)  # used different random_state
        self.assertNotEqual(gross_3, gross_2)  # used different random_state

    def test_get_expected_spectra(self):
        """Tests batch processing of seeds to expected counts.
        """
        target_counts = np.array([10, 100])

        # Single seed as 1-D array
        seed = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        expected_result = np.array([
            [1.,  1.,  1.,  1.,  1.,  1.],
            [10., 10., 10., 10., 10., 10.],
        ])
        result = get_expected_spectra(seed, target_counts)
        passed = np.array_equal(result, expected_result)
        self.assertTrue(passed, "Single seed as 1-D array failed.")

        # Multiple seeds in 2-D array
        seeds = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ])
        expected_result = np.array([
            [1.,  1.,  1.,  1.,  1.,  1.],
            [2.,  2.,  2.,  2.,  2.,  2.],
            [3.,  3.,  3.,  3.,  3.,  3.],
            [4.,  4.,  4.,  4.,  4.,  4.],
            [5.,  5.,  5.,  5.,  5.,  5.],
            [10., 10., 10., 10., 10., 10.],
            [20., 20., 20., 20., 20., 20.],
            [30., 30., 30., 30., 30., 30.],
            [40., 40., 40., 40., 40., 40.],
            [50., 50., 50., 50., 50., 50.],
        ])
        result = get_expected_spectra(seeds, target_counts)
        passed = np.array_equal(result, expected_result)
        self.assertTrue(passed, "Multiple seeds as 2-D array failed.")

    def test_get_expected_spectra_errors(self):
        """Tests certain checks when computing expected spectra."""
        invalid_target_counts = np.array([[10, 100]])  # because it's 2-D
        self.assertRaises(ValueError, get_expected_spectra, None, invalid_target_counts)

        invalid_target_counts = np.array([])  # because it's empty
        self.assertRaises(ValueError, get_expected_spectra, None, invalid_target_counts)

        valid_target_counts = np.array([10, 100])
        invalid_seeds = [
            (np.array([[[]]]), InvalidSeedError),  # because it's not 1-D or 2-D
        ]
        for invalid_seed, error in invalid_seeds:
            self.assertRaises(error, get_expected_spectra, invalid_seed, valid_target_counts)

    def test_get_merged_sources_samplewise(self):
        """Tests that sources DataFrames are properly merged."""
        # Adding identical DataFrames should result in the same source matrix times 2
        ss1 = get_dummy_seeds()
        ss2 = get_dummy_seeds()
        merged_sources_df = get_merged_sources_samplewise(ss1.sources, ss2.sources)
        expected_df = 2 * pd.DataFrame(
            np.identity(merged_sources_df.shape[0]),
            columns=ss1.sources.columns
        )
        self.assertTrue(
            merged_sources_df.equals(expected_df),
            "Merging identical DataFrames failed"
        )

        # Adding DataFrames with different columns should be doable
        sources1 = pd.DataFrame([
                [1.0, 0.0,  0.0],
                [0.0, 10.0, 0.0],
                [0.0, 5.0,  5.0],
            ],
            columns=["A", "B", "C"]
        )
        sources2 = pd.DataFrame([
                [0.0, 10.0, 0.0],
                [1.0, 5.0,  10.0],
                [1.0, 0.0,  0.0],
            ],
            columns=["C", "D", "E"]
        )
        expected_df = pd.DataFrame([
                [1.0, 0.0,  0.0, 10.0, 0.0],
                [0.0, 10.0, 1.0, 5.0,  10.0],
                [0.0, 5.0,  6.0, 0.0,  0.0],
            ],
            columns=["A", "B", "C", "D", "E"]
        )
        merged_sources_df = get_merged_sources_samplewise(sources1, sources2)
        self.assertTrue(
            merged_sources_df.equals(expected_df),
            "Merging overlapping DataFrames failed."
        )

    def test_get_samples_per_seed(self):
        """Tests label balancing at different levels."""
        seeds_ss = get_dummy_seeds()
        MIN_SPS = 1

        # Seed-level balancing
        seed_level_results, n_samples_expected = get_samples_per_seed(
            seeds_ss.sources.columns, MIN_SPS, "Seed"
        )
        self.assertTrue(all([x == MIN_SPS for x in seed_level_results.values()]))
        self.assertEqual(seeds_ss.n_samples, n_samples_expected)

        # Isotope-level balancing
        seed_level_results, n_samples_expected = get_samples_per_seed(
            seeds_ss.sources.columns, MIN_SPS, "Isotope"
        )
        self.assertEqual(seed_level_results["Am241"], 3)        # 3 samples
        self.assertEqual(seed_level_results["Ba133"], 3)        # 3 samples
        self.assertEqual(seed_level_results["K40"], 2)          # 4 samples
        self.assertEqual(seed_level_results["Ra226"], 3)        # 3 samples
        self.assertEqual(seed_level_results["Th232"], 3)        # 3 samples
        self.assertEqual(seed_level_results["U238"], 3)         # 3 samples
        self.assertEqual(seed_level_results["Pu239"], 1)        # 3 samples
        self.assertEqual(22, n_samples_expected)

        # Category-level balancing
        seed_level_results, n_samples_expected = get_samples_per_seed(
            seeds_ss.sources.columns, MIN_SPS, "Category"
        )
        self.assertEqual(seed_level_results["Industrial"], 2)   # 4 samples
        self.assertEqual(seed_level_results["NORM"], 1)         # 4 samples
        self.assertEqual(seed_level_results["SNM"], 1)          # 4 samples
        self.assertEqual(12, n_samples_expected)

    def test_get_batch(self):
        """Test batch ground truth."""
        synth = Synthesizer(
            bg_cps=300,
            apply_poisson_noise=False,
            return_fg=True,
            return_gross=True,
        )
        fg_seed = pd.Series([0.0, 0.0, 0.0, 0.2, 0.8])
        fg_sources = pd.Series(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            index=["A", "B", "C", "D", "E", "F", "G"]
        )
        bg_seed = pd.Series([0.2, 0.2, 0.6, 0.0, 0.0])
        bg_sources = pd.Series(
            [0.3, 0.4, 0.3],
            index=["X", "Y", "Z"]
        )
        lts = np.array([4.2]).astype(float)
        snrs = np.array([63.2]).astype(float)

        fg_ss, gross_ss = synth._get_batch(
            fg_seed=fg_seed,
            fg_sources=fg_sources,
            bg_seed=bg_seed,
            bg_sources=bg_sources,
            lt_targets=lts,
            snr_targets=snrs,
        )

        self.assertTrue(np.allclose(
            gross_ss.sources.loc[:, fg_sources.index].sum(axis=1)
            / np.sqrt(gross_ss.sources.loc[:, bg_sources.index].sum(axis=1)),
            snrs,
        ))
        self.assertTrue(np.allclose(
            gross_ss.sources.loc[:, bg_sources.index].sum(axis=1) / synth.bg_cps,
            lts,
        ))
        self.assertTrue(np.allclose(
            gross_ss.sources.loc[:, fg_sources.index],
            fg_ss.sources.loc[:, fg_sources.index],
        ))
        self.assertTrue(np.allclose(
            fg_ss.spectra.sum(axis=1),
            fg_ss.sources.loc[:, fg_sources.index].sum(axis=1),
        ))

    def test_source_proportions(self):
        """Test that SNR equals fg/bg source cumulative source proportions."""
        fg_seeds_ss, bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
        mixed_bg_seeds_ss = SeedMixer(bg_seeds_ss, mixture_size=3).generate(2)

        static_syn = StaticSynthesizer(
            samples_per_seed=1,
            snr_function="log10",
            rng=np.random.default_rng(1),
            return_fg=False,
            return_gross=True,
            apply_poisson_noise=False,
        )
        _, gross_ss = static_syn.generate(
            fg_seeds_ss=fg_seeds_ss,
            bg_seeds_ss=mixed_bg_seeds_ss,
            verbose=False,
        )

        fg_counts = gross_ss.sources.loc[:, fg_seeds_ss.sources.columns].sum(axis=1)\
            * gross_ss.info.total_counts
        bg_counts = gross_ss.sources.loc[:, bg_seeds_ss.sources.columns].sum(axis=1)\
            * gross_ss.info.total_counts
        self.assertTrue(np.allclose(
            fg_counts / np.sqrt(bg_counts),
            gross_ss.info.snr
        ))
