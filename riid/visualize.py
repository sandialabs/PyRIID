# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""This module provides visualization functions primarily for visualizing SampleSets."""
import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib import cm, rcParams  # noqa: E402
from matplotlib.colors import ListedColormap
from numpy.random import choice
from scipy.special import comb
from seaborn import heatmap
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
from sklearn.metrics import f1_score

from riid.sampleset import SampleSet

# DO NOT TOUCH what is set below, nor should you override them inside a function.
plt.style.use("default")
rcParams["font.family"] = "serif"
CM = cm.tab10
MARKER = "."


def save_or_show_plot(func):
    """Function decorator providing standardized handling of
    saving and/or showing matplotlib plots.

    Args:
        func: the function to call that builds the plot and
            returns a tuple of (Figure, Axes).
    """
    def save_or_show_plot_wrapper(*args, save_file_path=None, show=True,
                                  return_bytes=False, **kwargs):
        if return_bytes:
            matplotlib.use("Agg")
        fig, ax = func(*args, **kwargs)
        if save_file_path:
            fig.savefig(save_file_path)
        if show:
            plt.show()
        if save_file_path:
            plt.close(fig)
        if return_bytes:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            return buf
        return fig, ax

    return save_or_show_plot_wrapper


@save_or_show_plot
def confusion_matrix(ss: SampleSet, as_percentage: bool = False, cmap: str = "binary",
                     title: str = None, value_format: str = None, value_fontsize: int = None,
                     figsize: str = None, alpha: float = None):
    """Generates a confusion matrix for a SampleSet.

    Prediction and label information is used to distinguish between correct
    and incorrect classifications using color (green for correct, red for incorrect).

    Args:
        ss: a SampleSet of events to plot.
        as_percentage: scales existing confusion matrix values to the range 0 to 100.
        cmap: the colormap to use for seaborn colormap function.
        title: the plot title.
        value_format: the format string controlling how values are displayed in the matrix cells.
        value_fontsize: the font size of the values displayed in the matrix cells.
        figsize: the figure size passed to the matplotlib subplots call.
        alpha: the degree of opacity.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        EmptyPredictionsArrayError: raised when the sampleset does not contain any predictions
    """
    y_true = ss.labels
    y_pred = ss.predictions
    labels = sorted(set(list(y_true) + list(y_pred)))

    if y_pred.size == 0:
        msg = "Predictions array was empty.  Have you called `model.predict(ss)`?"
        raise EmptyPredictionsArrayError(msg)

    if not cmap:
        cmap = ListedColormap(['white'])

    cm_values = confusion_matrix_sklearn(y_true, y_pred, labels=labels)
    if as_percentage:
        cm_values = np.array(cm_values)
        cm_values = cm_values / cm_values.sum(axis=1)
        if not value_format:
            value_format = ".1%"
    else:
        if not value_format:
            value_format = ".0f"

    heatmap_kwargs = {}
    if alpha:
        heatmap_kwargs.update({"alpha": alpha})
    if value_format:
        heatmap_kwargs.update({"fmt": value_format})
    if cmap:
        heatmap_kwargs.update({"cmap": cmap})

    fig, ax = plt.subplots(**{"figsize": figsize})
    mask = cm_values == 0
    ax = heatmap(cm_values, annot=True, linewidths=0.25, linecolor="grey", cbar=False,
                 mask=mask, **heatmap_kwargs)

    tick_locs = np.arange(len(labels)) + 0.5
    ax.set_ylabel("Truth")
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(labels, rotation=0)

    ax.set_xlabel("Prediction")
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(labels, rotation=90)

    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


@save_or_show_plot
def plot_live_time_vs_snr(ss: SampleSet, overlay_ss: SampleSet = None, alpha: float = 0.5,
                          xscale: str = "linear", yscale: str = "log",
                          xlim: tuple = None, ylim: tuple = None,
                          title: str = "Live Time vs. SNR", sigma_line_value: float = None):
    """Plots SNR against live time for all samples in a SampleSet.

    Prediction and label information is used to distinguish between correct
    and incorrect classifications using color (green for correct, red for incorrect).

    Args:
        ss: a SampleSet of events to plot.
        overlay_ss: another SampleSet to color as black.
        alpha: the degree of opacity (not applied to overlay_ss scatterplot if used).
        xscale: the X-axis scale.
        yscale: the Y-axis scale.
        xlim: a tuple containing the X-axis min and max values.
        ylim: a tuple containing the Y-axis min and max values.
        title: the plot title.
        sigma_line_value: plots a sigma line representing the `value` number of
            standard deviations from background.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    correct_ss = ss.get_indices(ss.labels == ss.predictions)
    incorrect_ss = ss.get_indices(ss.labels != ss.predictions)
    if not xlim:
        xlim = (ss.live_time.min(), ss.live_time.max())
    if not ylim:
        if yscale == "log":
            ylim = (ss.snr_estimate.clip(1e-3).min(), ss.snr_estimate.max())
        else:
            ylim = (ss.snr_estimate.clip(0).min(), ss.snr_estimate.max())
    fig, ax = plt.subplots()

    ax.scatter(
        correct_ss.live_time,
        correct_ss.snr_estimate,
        c="green", alpha=alpha, marker=MARKER, label="Correct"
    )
    ax.scatter(
        incorrect_ss.live_time,
        incorrect_ss.snr_estimate,
        c="red", alpha=alpha, marker=MARKER, label="Incorrect"
    )
    if overlay_ss:
        plt.scatter(
            overlay_ss.live_time,
            overlay_ss.snr_estimate,
            c="black", marker="+", label="Event" + ("" if overlay_ss.n_samples == 1 else "s"),
            s=75
        )
    if sigma_line_value:
        live_times = np.linspace(xlim[0], xlim[1])
        background_cps = ss.collection_information["bg_counts_expected"][0] / ss.live_time[0]
        snrs = sigma_line_value / np.sqrt(live_times * background_cps)
        plt.plot(
            live_times,
            snrs,
            c="blue",
            alpha=alpha,
            label="{}-sigma".format(sigma_line_value),
            ls="dashed"
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Live Time (s)")
    ax.set_ylabel("Signal-to-Noise Ratio (SNR)")
    ax.set_title(title)
    fig.legend(loc="lower right")
    return fig, ax


@save_or_show_plot
def plot_strength_vs_score(ss: SampleSet, overlay_ss: SampleSet = None, alpha: float = 0.5,
                           marker_size=75, xscale: str = "log", yscale: str = "linear",
                           xlim: tuple = (None, None), ylim: tuple = (0, 1.05),
                           title: str = "Signal Strength vs. Score", sigma_line_value: float = None):
    """Plots strength against prediction score for all samples in a SampleSet.

    Prediction and label information is used to distinguish between correct
    and incorrect classifications using color (green for correct, red for incorrect).

    Args:
        ss: a SampleSet of events to plot.
        overlay_ss: another SampleSet to color as blue (correct) and/or black (incorrect).
        alpha: the degree of opacity (not applied to overlay_ss scatterplot if used).
        xscale: the X-axis scale.
        yscale: the Y-axis scale.
        xlim: a tuple containing the X-axis min and max values.
        ylim: a tuple containing the Y-axis min and max values.
        title: the plot title.
        sigma_line_value: plots a sigma line representing the `value` number of
            standard deviations from background.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    correct_ss = ss.get_indices(ss.labels == ss.predictions)
    incorrect_ss = ss.get_indices(ss.labels != ss.predictions)
    if not xlim:
        if xscale == "log":
            xlim = (ss.sigma.clip(1e-3).min(), ss.sigma.max())
        else:
            xlim = (ss.sigma.clip(0).min(), ss.sigma.max())
    fig, ax = plt.subplots()

    ax.scatter(
        correct_ss.sigma,
        correct_ss.prediction_probas.max(axis=1),
        c="green", alpha=alpha, marker=MARKER, label="Correct", s=marker_size
    )
    ax.scatter(
        incorrect_ss.sigma,
        incorrect_ss.prediction_probas.max(axis=1),
        c="red", alpha=alpha, marker=MARKER, label="Incorrect", s=marker_size
    )
    if overlay_ss:
        overlay_correct_ss = overlay_ss.get_indices(overlay_ss.labels == overlay_ss.predictions)
        overlay_incorrect_ss = overlay_ss.get_indices(overlay_ss.labels != overlay_ss.predictions)
        ax.scatter(
            overlay_correct_ss.sigma,
            overlay_correct_ss.prediction_probas.max(axis=1),
            c="blue", marker="*", label="Correct Event" + ("" if overlay_incorrect_ss.n_samples == 1 else "s"),
            s=marker_size*1.25
        )
        ax.scatter(
            overlay_incorrect_ss.sigma,
            overlay_incorrect_ss.prediction_probas.max(axis=1),
            c="black", marker="+", label="Incorrect Event" + ("" if overlay_incorrect_ss.n_samples == 1 else "s"),
            s=marker_size*1.25
        )
    if sigma_line_value:
        ax.vlines(
            sigma_line_value,
            0,
            1,
            colors="blue",
            alpha=alpha,
            label="{}-sigma".format(sigma_line_value),
            linestyles="dashed"
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Sigma (net / sqrt(background))")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    return fig, ax


@save_or_show_plot
def plot_snr_vs_score(ss: SampleSet, overlay_ss: SampleSet = None, alpha: float = 0.5,
                      marker_size=75, xscale: str = "log", yscale: str = "linear",
                      xlim: tuple = (None, None), ylim: tuple = (0, 1.05),
                      title: str = "SNR vs. Score"):
    """Plots SNR against prediction score for all samples in a SampleSet.

    Prediction and label information is used to distinguish between correct
    and incorrect classifications using color (green for correct, red for incorrect).

    Args:
        ss: a SampleSet of events to plot.
        overlay_ss: another SampleSet to color as blue (correct) and/or black (incorrect).
        alpha: the degree of opacity (not applied to overlay_ss scatterplot if used).
        xscale: the X-axis scale.
        yscale: the Y-axis scale.
        xlim: a tuple containing the X-axis min and max values.
        ylim: a tuple containing the Y-axis min and max values.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    correct_ss = ss.get_indices(ss.labels == ss.predictions)
    incorrect_ss = ss.get_indices(ss.labels != ss.predictions)
    if not xlim:
        if xscale == "log":
            xlim = (ss.snr_estimate.clip(1e-3).min(), ss.snr_estimate.max())
        else:
            xlim = (ss.snr_estimate.clip(0).min(), ss.snr_estimate.max())
    fig, ax = plt.subplots()

    ax.scatter(
        correct_ss.snr_estimate,
        correct_ss.prediction_probas.max(axis=1),
        c="green", alpha=alpha, marker=MARKER, label="Correct", s=marker_size
    )
    ax.scatter(
        incorrect_ss.snr_estimate,
        incorrect_ss.prediction_probas.max(axis=1),
        c="red", alpha=alpha, marker=MARKER, label="Incorrect", s=marker_size
    )
    if overlay_ss:
        overlay_correct_ss = overlay_ss.get_indices(overlay_ss.labels == overlay_ss.predictions)
        overlay_incorrect_ss = overlay_ss.get_indices(overlay_ss.labels != overlay_ss.predictions)
        ax.scatter(
            overlay_correct_ss.snr_estimate,
            overlay_correct_ss.prediction_probas.max(axis=1),
            c="blue", marker="*", label="Correct Event" + ("" if overlay_correct_ss.n_samples == 1 else "s"),
            s=marker_size*1.25
        )
        ax.scatter(
            overlay_incorrect_ss.snr_estimate,
            overlay_incorrect_ss.prediction_probas.max(axis=1),
            c="black", marker="+", label="Incorrect Event" + ("" if overlay_incorrect_ss.n_samples == 1 else "s"),
            s=marker_size*1.25
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("SNR (net / background)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()

    return fig, ax


@save_or_show_plot
def plot_spectra(ss: SampleSet, is_in_energy: bool = False, limit: int = None,
                 figsize: tuple = None, xscale: str = "linear", yscale: str = "log",
                 xlim: tuple = (0, None), ylim: tuple = (1e-1, None),
                 ylabel: str = None, title: str = None, legend_loc: str = None) -> tuple:
    """Plots the spectra contained with a SampleSet.

    Args:
        ss: spectra to plot.
        is_in_energy: whether or not to try and use each spectrum's
            energy bin values to convert the spectrum from bins to energy.
        limit: the number of spectra to plot; None will plot all.
        figsize: the figure size passed to the matplotlib subplots call.
        xscale: the X-axis scale.
        yscale: the Y-axis scale.
        xlim: a tuple containing the X-axis min and max values.
        ylim: a tuple containing the Y-axis min and max values.
        ylabel: the Y-axis label.
        title: the plot title.
        legend_loc: the location in which to place the legend. Defaults to None.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        ValueError: is_in_energy=True but energy bin centers are missing for any spectra.
        ValueError: limit is not None and less than 1.
    """
    if is_in_energy and pd.isnull(ss.energy_bin_centers.reshape(-1)).any():
        msg = "When using 'is_in_energy' a valid energy calibration is required."
        raise ValueError(msg)
    if limit and limit < 1:
        raise ValueError("'limit' argument can not be less than 1.")

    if limit:
        ss = ss.get_indices(range(limit))
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(ss.spectra.index)):
        label = ss.labels[i]

        if is_in_energy:
            xvals = ss.energy_bin_centers[i, :]
        else:
            xvals = np.arange(ss.n_channels)
        ax.plot(
            xvals,
            ss.spectra.iloc[i],
            label=label,
            color=CM(i),
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if is_in_energy:
        if xscale == "log":
            ax.set_xlabel("log(Energy (keV))")
        else:
            ax.set_xlabel("Energy (keV)")
    else:
        if xscale == "log":
            ax.set_xlabel("log(Channel)")
        else:
            ax.set_xlabel("Channel")
    if ylabel:
        ax.set_ylabel(ylabel)
    elif yscale == "log":
        ax.set_ylabel("log(Counts)")
    else:
        ax.set_ylabel("Counts")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Gamma Spectr" + ("um" if ss.n_samples == 1 else "a"))
    if legend_loc:
        ax.legend(loc=legend_loc)
    else:
        ax.legend()

    return fig, ax


@save_or_show_plot
def plot_learning_curve(train_loss: list, validation_loss: list,
                        xscale: str = "linear", yscale: str = "linear",
                        xlim: tuple = (0, None), ylim: tuple = (0, None),
                        ylabel: str = "Loss", legend_loc: str = "upper right",
                        smooth: bool = False, title: str = None) -> tuple:
    """Plots training and validation loss curves.

    Args:
        train_loss: list of training loss values.
        validation_loss: list of validation loss values.
        xscale: the X-axis scale.
        yscale: the Y-axis scale.
        xlim: a tuple containing the X-axis min and max values.
        ylim: a tuple containing the Y-axis min and max values.
        smooth: whether or not to apply smoothing to the loss curves.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        ValueError:
            - if either list of values is empty
    """
    train_loss = np.array(train_loss)
    validation_loss = np.array(validation_loss)
    if train_loss.size == 0:
        raise ValueError("List of training loss values was not provided.")
    if validation_loss.size == 0:
        raise ValueError("List of validation loss values was not provided.")

    if isinstance(train_loss[0], (list, tuple)):
        train_x = np.array([ep for ep, _ in train_loss])
        train_y = np.array([lv for _, lv in train_loss])
    else:
        train_x = np.arange(len(train_loss))
        train_y = np.array([lv for lv in train_loss])

    if isinstance(validation_loss[0], (list, tuple)):
        val_x = np.array([ep for ep, _ in validation_loss])
        val_y = np.array([lv for _, lv in validation_loss])
    else:
        val_x = np.arange(len(validation_loss))
        val_y = np.array([lv for lv in validation_loss])

    fig, ax = plt.subplots()
    if smooth:
        from scipy.interpolate import make_interp_spline

        # The 300 one the next line is the number of points to make between min and max
        train_xnew = np.linspace(train_x.min(), train_x.max(), 50)
        spl = make_interp_spline(train_x, train_y, k=3)
        train_ps = spl(train_xnew)

        val_xnew = np.linspace(val_x.min(), val_x.max(), 300)
        spl = make_interp_spline(val_x, val_y, k=3)
        val_ps = spl(val_xnew)

        ax.plot(train_xnew, train_ps, label="Train", color=CM(0))
        ax.plot(val_xnew, val_ps, label="Validation", color=CM(1))
        ax.hlines(train_ps[-1], xlim[0], train_x.max(), color=CM(0), linestyles="dashed")
        ax.hlines(val_ps[-1], xlim[0], val_x.max(), color=CM(1), linestyles="dashed")
    else:
        ax.plot(train_x, train_y, label="Train", color=CM(0))
        ax.plot(val_x, val_y, label="Validation", color=CM(1))
        ax.hlines(train_y[-1], xlim[0], val_x.max(), color=CM(0), linestyles="dashed")
        ax.hlines(val_y[-1], xlim[0], val_x.max(), color=CM(1), linestyles="dashed")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Learning Curve")
    ax.legend(loc=legend_loc)

    return fig, ax


@save_or_show_plot
def plot_count_rate_history(cr_history: list, sample_interval: float,
                            event_duration: float, pre_event_duration: float,
                            ylim: tuple = (0, None), title: str = None):
    """Plots a count rate history.

    Args:
        cr_history: list of count rate values.
        sample_interval: the time in seconds for which each count rate values was collected.
        event_duration: the time in seconds during which an anomalous source was present.
        pre_event_duration: the time in seconds at which the anomalous source appears
            (i.e., the start of the event).
        validation_loss: list of validation loss values.
        ylim: a tuple containing the Y-axis min and max values.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    fig, ax = plt.subplots()

    time_steps = np.arange(
        start=-pre_event_duration,
        stop=len(cr_history) * sample_interval - pre_event_duration,
        step=sample_interval
    )
    ax.plot(
        time_steps,
        cr_history,
        color=CM(0)
    )
    ax.axvspan(
        xmin=0,
        xmax=event_duration,
        facecolor=CM(0),
        alpha=0.1
    )
    ax.set_ylim(ylim)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Counts per second")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Count Rate History")

    return fig, ax


@save_or_show_plot
def plot_score_histogram(ss: SampleSet, yscale="log", ylim=(1e-1, None),
                         title="Score Distribution"):
    """Plots a histogram of all of the model prediction scores.

    Args:
        ss: SampleSet containing prediction_probas values.
        yscale: the Y-axis scale.
        ylim: a tuple containing the Y-axis min and max values.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    fig, ax = plt.subplots()

    indices1 = ss.collection_information.index[ss.collection_information["sigma"] <= 5]
    values1 = ss.prediction_probas.loc[indices1].values.flatten()
    values1 = np.where(values1 > 0.0, values1, values1)
    indices2 = ss.collection_information.index[(ss.collection_information["sigma"] > 5) &
                                               (ss.collection_information["sigma"] <= 50)]
    values2 = ss.prediction_probas.loc[indices2].values.flatten()
    values2 = np.where(values2 > 0.0, values2, values2)
    indices3 = ss.collection_information.index[ss.collection_information["sigma"] > 50]
    values3 = ss.prediction_probas.loc[indices3].values.flatten()
    values3 = np.where(values3 > 0.0, values3, values3)

    width = 0.35
    bins = np.linspace(0.0, 1.0, 100)
    ax.bar(
        values3,
        bins,
        width,
        color="green"
    )
    ax.bar(
        values2,
        bins,
        width,
        color="yellow"
    )
    ax.bar(
        values1,
        bins,
        width,
        color="red"
    )

    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    ax.set_xlabel("Scores")
    ax.set_ylabel("Occurrences")
    ax.set_title(title)

    return fig, ax


@save_or_show_plot
def plot_n_isotopes_vs_f1_bayes(ss: SampleSet, seeds: SampleSet, title="Number of Isotopes vs. F1 Score"):
    """Plots the F1 score for different numbers of isotopes under consideration,
    specifically for a PoissonBayes model.

    Args:
        ss: SampleSet containing prediction_probas values.
        seeds: the same seeds that were used by the PoissonBayes model.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    n_to_sample = 20
    isotopes = np.array(list(set(seeds.labels)))
    isotope_to_index = {isotope: [] for isotope in isotopes}

    for i in range(seeds.labels.shape[0]):
        isotope_to_index[seeds.labels[i]].append(i)

    f1_scores = np.zeros((isotopes.shape[0] - 1,))
    for n in range(1, isotopes.shape[0]):
        inds = [choice(isotopes.shape[0], n + 1, replace=False) for _ in
                range(min(n_to_sample, comb(isotopes.shape[0], n, exact=True)))]
        for j in inds:
            # the isotopes we are considering this iteration
            curr_isotopes = isotopes[j]
            assert len(j) == n + 1
            proba_indicies = []
            for iso in curr_isotopes:
                proba_indicies += isotope_to_index[iso]
            proba_indicies.sort()

            # get the isotopes, whose correct label is in the set of isotopes we are considering
            i_labels = range(ss.labels.shape[0])
            all_proba_indicies = [i for i in i_labels if ss.labels[i] in curr_isotopes]
            # get probas that we need
            curr_probas = ss.prediction_probas.values[all_proba_indicies]
            curr_probas = curr_probas[:, proba_indicies]
            max_indicies = curr_probas.argmax(axis=1)
            predictions = [seeds.labels[proba_indicies[i]] for i in max_indicies]
            labels = ss.labels[all_proba_indicies]
            f1_scores[n - 1] += (f1_score(labels, predictions, average="micro"))

        f1_scores[n - 1] /= len(inds)

    fig, ax = plt.subplots()
    plt.plot(np.arange(2, f1_scores.shape[0] + 2), f1_scores)
    ax.set_ylim((0, 1.1))
    ax.set_xlabel("Number of Isotopes")
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    return fig, ax


@save_or_show_plot
def plot_n_isotopes_vs_f1(f1_scores: list, title: str = "Number of Isotopes vs. F1 Score"):
    """Plots the pre-computed F1 score for different numbers of isotopes under consideration.

    Args:
        f1_scores: list of pre-computed F1 scores.
        title: the plot title.

    Returns:
        A tuple (Figure, Axes) of matplotlib objects.

    Raises:
        None
    """
    fig, ax = plt.subplots()
    ax.plot([x for x in range(1, len(f1_scores)+1)], f1_scores)
    ax.set_ylim((0, 1.1))
    ax.set_xlabel("Number of Isotopes")
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    return fig, ax


class EmptyPredictionsArrayError(Exception):
    pass
