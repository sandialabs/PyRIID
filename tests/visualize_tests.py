# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module tests the visualize module."""
import unittest

import numpy as np

from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.metrics import precision_recall_curve
from riid.models.neural_nets import MLPClassifier
from riid.visualize import (plot_correlation_between_all_labels,
                            plot_count_rate_history,
                            plot_label_and_prediction_distributions,
                            plot_label_distribution, plot_learning_curve,
                            plot_live_time_vs_snr, plot_precision_recall,
                            plot_prediction_distribution,
                            plot_score_distribution, plot_snr_vs_score,
                            plot_spectra, plot_ss_comparison)


class TestVisualize(unittest.TestCase):
    """Testing plot functions in the visualize module."""
    def setUp(self):
        """Test setup."""
        self.fg_seeds_ss, self.bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
        self.mixed_bg_seed_ss = SeedMixer(self.bg_seeds_ss, mixture_size=3).generate(10)

        self.static_synth = StaticSynthesizer(
            samples_per_seed=100,
            snr_function="log10",
            return_fg=False,
            return_gross=True
        )
        _, _, self.train_ss = self.static_synth.generate(
            self.fg_seeds_ss,
            self.mixed_bg_seed_ss,
            verbose=False
        )
        self.train_ss.normalize()

        model = MLPClassifier()
        self.history = model.fit(self.train_ss, epochs=10, patience=5).history
        model.predict(self.train_ss)

        # Generate some test data
        self.static_synth.samples_per_seed = 50
        _, _, self.test_ss = self.static_synth.generate(
            self.fg_seeds_ss,
            self.mixed_bg_seed_ss,
            verbose=False
        )
        self.test_ss.normalize()
        model.predict(self.test_ss)

    def test_plot_live_time_vs_snr(self):
        """Plots SNR against live time for all samples in a SampleSet.
        Prediction and label information is used to distinguish between correct and incorrect
        classifications using color (green for correct, red for incorrect).
        """
        plot_live_time_vs_snr(self.test_ss, show=False)
        plot_live_time_vs_snr(self.train_ss, self.test_ss, show=False)

    def test_plot_snr_vs_score(self):
        """Plots SNR against prediction score for all samples in a SampleSet.
        Prediction and label information is used to distinguish between correct and incorrect
        classifications using color (green for correct, red for incorrect).
        """
        plot_snr_vs_score(self.train_ss, show=False)
        plot_snr_vs_score(self.train_ss, self.test_ss, show=False)

    def test_plot_spectra(self):
        """Plots the spectra contained with a SampleSet."""
        plot_spectra(self.fg_seeds_ss, ylim=(None, None), in_energy=False, show=False)
        plot_spectra(self.fg_seeds_ss, ylim=(None, None), in_energy=True, show=False)

    def test_plot_learning_curve(self):
        """Plots training and validation loss curves."""
        plot_learning_curve(self.history["loss"],
                            self.history["val_loss"],
                            show=False)
        plot_learning_curve(self.history["loss"],
                            self.history["val_loss"],
                            smooth=True,
                            show=False)

    def test_plot_count_rate_history(self):
        """Plots a count rate history."""
        counts = np.random.normal(size=1000)
        histogram, _ = np.histogram(counts, bins=100, range=(0, 100))
        plot_count_rate_history(histogram, 1, 80, 20, show=False)

    def test_plot_label_and_prediction_distributions(self):
        """Plots distributions of data labels, predictions, and prediction scores."""
        plot_score_distribution(self.test_ss, show=False)
        plot_label_distribution(self.test_ss, show=False)
        plot_prediction_distribution(self.test_ss, show=False)
        plot_label_and_prediction_distributions(self.test_ss, show=False)

    def test_plot_correlation_between_all_labels(self):
        """Plots a correlation matrix of each label against each other label."""
        plot_correlation_between_all_labels(self.bg_seeds_ss, show=False)
        plot_correlation_between_all_labels(self.bg_seeds_ss, mean=True, show=False)

    def test_plot_precision_recall(self):
        """Plots the multi-class or multi-label Precision-Recall curve and marks the optimal
        F1 score for each class.
        """
        precision, recall, _ = precision_recall_curve(self.test_ss)
        plot_precision_recall(precision=precision, recall=recall, show=False)

    def test_plot_ss_comparison(self):
        """Creates a plot for output from SampleSet.compare_to()."""
        SYNTHETIC_DATA_CONFIG = {
            "samples_per_seed": 100,
            "bg_cps": 100,
            "snr_function": "uniform",
            "snr_function_args": (1, 100),
            "live_time_function": "uniform",
            "live_time_function_args": (0.25, 10),
            "apply_poisson_noise": True,
            "return_fg": False,
            "return_gross": True,
        }

        _, _, gross_ss1 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
            .generate(self.fg_seeds_ss, self.mixed_bg_seed_ss, verbose=False)
        _, _, gross_ss2 = StaticSynthesizer(**SYNTHETIC_DATA_CONFIG)\
            .generate(self.fg_seeds_ss, self.mixed_bg_seed_ss, verbose=False)

        # Compare two different gross sample sets - Live time
        ss1_stats, ss2_stats, col_comparisons = gross_ss1.compare_to(gross_ss2,
                                                                     density=False)
        plot_ss_comparison(ss1_stats,
                           ss2_stats,
                           col_comparisons,
                           "live_time",
                           show=False)

        # Compare two different gross sample sets - Total Counts
        ss1_stats, ss2_stats, col_comparisons = gross_ss1.compare_to(gross_ss2,
                                                                     density=True)
        plot_ss_comparison(ss1_stats,
                           ss2_stats,
                           col_comparisons,
                           "total_counts",
                           show=False)

        # Compare the same sampleset to itself - Live time
        ss1_stats, ss2_stats, col_comparisons = gross_ss1.compare_to(gross_ss1,
                                                                     density=False)
        plot_ss_comparison(ss1_stats,
                           ss2_stats,
                           col_comparisons,
                           "live_time",
                           show=False)

        # Compare the same sampleset to itself - Total Counts
        ss1_stats, ss2_stats, col_comparisons = gross_ss2.compare_to(gross_ss2,
                                                                     density=True)
        plot_ss_comparison(ss1_stats,
                           ss2_stats,
                           col_comparisons,
                           "total_counts",
                           show=False)


if __name__ == "__main__":
    unittest.main()
