# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module provides visualization functions primarily for visualizing SampleSets."""
import hashlib
from functools import wraps
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib import cm, rcParams  # noqa: E402
from matplotlib.colors import ListedColormap
from seaborn import heatmap
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn

from riid.data import SampleSet

# DO NOT TOUCH what is set below nor override them inside a function.
plt.style.use("default")
rcParams["font.family"] = "serif"
CM = cm.tab20
MARKER = "."


def save_or_show_plot(func):
    """Function decorator standardizing handling of saving and/or showing matplotlib plots.

    Args:
        func: function to call that builds the plot and returns a tuple of (Figure, Axes)
    """
    @wraps(func)
    def save_or_show_plot_wrapper(*args, save_file_path=None, show=True,
                                  return_bytes=False, **kwargs):
        if return_bytes:
            matplotlib.use("Agg")
        fig, ax = func(*args, **kwargs)
        plt.tight_layout()
        if save_file_path:
            fig.savefig(save_file_path)
        if show:
            plt.show()
        if save_file_path:
            plt.close(fig)
        if return_bytes:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            plt.close(fig)
            return buf
        return fig, ax

    return save_or_show_plot_wrapper


@save_or_show_plot
def confusion_matrix(ss: SampleSet, as_percentage: bool = False, cmap: str = "binary",
                     title: str = None, value_format: str = None, value_fontsize: int = None,
                     figsize=(10, 10), alpha: float = None, target_level="Isotope"):
    """Generate a confusion matrix for a SampleSet.

    Args:
        ss: `SampleSet` of events to plot
        as_percentage: scales existing confusion matrix values to the range 0 to 100
        cmap: colormap to use for seaborn colormap function
        title: plot title
        value_format: format string controlling how values are displayed in the matrix cells
        value_fontsize: font size of the values displayed in the matrix cells
        figsize: with and height of figure in inches
        alpha: degree of opacity
        target_level: `SampleSet.sources` column level to use

    Returns:
        Tuple (Figure, Axes) of matplotlib objects

    Raises:
        `EmptyPredictionsArrayError` when the `SampleSet` does not contain any predictions
    """
    y_true = ss.get_labels(target_level=target_level)
    y_pred = ss.get_predictions(target_level=target_level)
    labels = sorted(set(list(y_true) + list(y_pred)))

    if y_pred.size == 0:
        msg = "Predictions array was empty.  Have you called `model.predict(ss)`?"
        raise EmptyPredictionsArrayError(msg)

    if not cmap:
        cmap = ListedColormap(["white"])

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

    fig, ax = plt.subplots(figsize=figsize)
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

    return fig, ax


@save_or_show_plot
def plot_live_time_vs_snr(ss: SampleSet, overlay_ss: SampleSet = None, alpha: float = 0.5,
                          xscale: str = "linear", yscale: str = "log",
                          xlim: tuple = None, ylim: tuple = None,
                          title: str = "Live Time vs. SNR", snr_line_value: float = None,
                          figsize=(6.4, 4.8), target_level: str = "Isotope"):
    """Plot `SampleSet.info.snr` against `SampleSet.info.live_time`.

    Prediction and label information is used to distinguish between correct and incorrect
    classifications using color (blue for correct, red for incorrect).

    Args:
        ss: `SampleSet` of events to plot
        overlay_ss: another `SampleSet` to color as black
        alpha: degree of opacity (not applied to overlay_ss scatterplot if used)
        xscale: x-axis scale
        yscale: y-axis scale
        xlim: tuple containing the x-axis min and max values
        ylim: tuple containing the y-axis min and max values
        title: plot title
        snr_line_value: Plots a vertical line for contextualizing data to threshold
        figsize: with and height of figure in inches
        target_level: `SampleSet.sources` column level to use

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    labels = ss.get_labels(target_level=target_level)
    predictions = ss.get_predictions(target_level=target_level)
    correct_ss = ss[labels == predictions]
    incorrect_ss = ss[labels != predictions]
    if not xlim:
        xlim = (ss.info.live_time.min(), ss.info.live_time.max())
    if not ylim:
        if yscale == "log":
            ylim = (ss.info.snr.clip(1e-3).min(), ss.info.snr.max())
        else:
            ylim = (ss.info.snr.clip(0).min(), ss.info.snr.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        correct_ss.info.live_time,
        correct_ss.info.snr,
        c="blue", alpha=alpha, marker=MARKER, label="Correct"
    )
    ax.scatter(
        incorrect_ss.info.live_time,
        incorrect_ss.info.snr,
        c="red", alpha=alpha, marker=MARKER, label="Incorrect"
    )
    if overlay_ss:
        plt.scatter(
            overlay_ss.info.live_time,
            overlay_ss.info.snr,
            c="black", marker="+", label="Event" + ("" if overlay_ss.n_samples == 1 else "s"),
            s=75
        )
    if snr_line_value:
        live_times = np.linspace(xlim[0], xlim[1])
        plt.plot(
            live_times,
            snr_line_value,
            c="black",
            alpha=alpha,
            label=f"SNR={snr_line_value}",
            ls="dashed"
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Live Time (s)")
    ax.set_ylabel("Signal-to-Noise Ratio (SNR)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax


@save_or_show_plot
def plot_snr_vs_score(ss: SampleSet, overlay_ss: SampleSet = None, alpha: float = 0.5,
                      marker_size=75, xscale: str = "log", yscale: str = "linear",
                      xlim: tuple = (None, None), ylim: tuple = (0, 1.05),
                      title: str = "SNR vs. Score", figsize=(6.4, 4.8), target_level="Isotope"):
    """Plot `SampleSet.info.snr` against `SampleSet.prediction_probas`.

    Prediction and label information is used to distinguish between correct and incorrect
    classifications using color (blue for correct, red for incorrect).

    Args:
        ss: `SampleSet` of events to plot
        overlay_ss: another `SampleSet` to color as blue (correct) and/or black (incorrect)
        alpha: degree of opacity (not applied to overlay_ss scatterplot if used)
        xscale: x-axis scale
        yscale: y-axis scale
        xlim: tuple containing the x-axis min and max values
        ylim: tuple containing the y-axis min and max values
        title: plot title
        figsize: with and height of figure in inches
        target_level: `SampleSet.sources` column level to use

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    labels = ss.get_labels(target_level=target_level)
    predictions = ss.get_predictions(target_level=target_level, level_aggregation=None)
    correct_ss = ss[labels == predictions]
    incorrect_ss = ss[labels != predictions]
    if not xlim:
        if xscale == "log":
            xlim = (ss.info.snr.clip(1e-3).min(), ss.info.snr.max())
        else:
            xlim = (ss.info.snr.clip(0).min(), ss.info.snr.max())

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        correct_ss.info.snr,
        correct_ss.prediction_probas.max(axis=1),
        c="blue", alpha=alpha, marker=MARKER, label="Correct", s=marker_size
    )
    ax.scatter(
        incorrect_ss.info.snr,
        incorrect_ss.prediction_probas.max(axis=1),
        c="red", alpha=alpha, marker=MARKER, label="Incorrect", s=marker_size
    )
    if overlay_ss:
        overlay_labels = overlay_ss.get_labels()
        overlay_predictions = overlay_ss.get_predictions()
        overlay_correct_ss = overlay_ss[overlay_labels == overlay_predictions]
        overlay_incorrect_ss = overlay_ss[overlay_labels != overlay_predictions]
        ax.scatter(
            overlay_correct_ss.info.snr,
            overlay_correct_ss.prediction_probas.max(axis=1),
            c="purple",
            marker="*",
            label="Correct Event" + ("" if overlay_correct_ss.n_samples == 1 else "s"),
            s=marker_size*1.25
        )
        ax.scatter(
            overlay_incorrect_ss.info.snr,
            overlay_incorrect_ss.prediction_probas.max(axis=1),
            c="yellow",
            marker="+",
            label="Incorrect Event" + ("" if overlay_incorrect_ss.n_samples == 1 else "s"),
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
def plot_spectra(ss: SampleSet, in_energy: bool = False,
                 figsize: tuple = (12.8, 7.2), xscale: str = "linear", yscale: str = "log",
                 xlim: tuple = (None, None), ylim: tuple = (None, None),
                 ylabel: str = None, title: str = None, legend_loc: str = None,
                 target_level="Isotope") -> tuple:
    """Plot spectra in a `SampleSet`.

    Args:
        ss: `SampleSet` with spectra to plot
        in_energy: whether to try and use each spectrum's e-cal to display bin energy
        figsize: width and height of figure in inches
        xscale: x-axis scale
        yscale: y-axis scale
        xlim: tuple containing the x-axis min and max values
        ylim: tuple containing the y-axis min and max values
        ylabel: y-axis label
        title: plot title
        legend_loc: location in which to place the legend
        target_level: `SampleSet.sources` column level to use in legend

    Returns:
        Tuple (Figure, Axes) of matplotlib objects

    Raises:
        `ValueError` when:

        - `is_in_energy` equals True but energy bin centers are missing for any spectra
        - `limit` is not None and less than 1
    """
    fig, ax = plt.subplots(figsize=figsize)
    if ss.sources.empty:
        labels = list(range(ss.n_samples))
    else:
        labels = ss.get_labels(target_level=target_level)

    for i in range(ss.n_samples):
        label = labels[i]
        if in_energy:
            xvals = ss.get_channel_energies(i)
        else:
            xvals = np.arange(ss.n_channels)
        ax.plot(
            xvals,
            ss.spectra.iloc[i],
            label=label,
            color=CM(i % CM.N),
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if in_energy:
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
                        smooth: bool = False, title: str = None, figsize=(6.4, 4.8)) -> tuple:
    """Plot training and validation loss curves.

    Args:
        train_loss: list of training loss values
        validation_loss: list of validation loss values
        xscale: x-axis scale
        yscale: y-axis scale
        xlim: tuple containing the x-axis min and max values
        ylim: tuple containing the y-axis min and max values
        smooth: whether to apply smoothing to the loss curves
        title: plot title
        figsize: with and height of figure in inches

    Returns:
        Tuple (Figure, Axes) of matplotlib objects

    Raises:
        `ValueError` when either list of values is empty
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

    fig, ax = plt.subplots(figsize=figsize)
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
                            ylim: tuple = (0, None), title: str = None, figsize=(6.4, 4.8)):
    """Plot a count rate history.

    Args:
        cr_history: list of count rate values
        sample_interval: time in seconds for which each count rate values was collected
        event_duration: time in seconds during which an anomalous source was present
        pre_event_duration: time in seconds at which the anomalous source appear
            (i.e., the start of the event)
        validation_loss: list of validation loss values
        ylim: tuple containing the y-axis min and max values
        title: plot title
        figsize: width and height of figure in inches

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

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
def plot_score_distribution(ss: SampleSet, bin_width=None, n_bins=100,
                            xscale="linear", min_bin=0.0, max_bin=1.0,
                            yscale="log", ylim=(1e-1, None),
                            title="Score Distribution", figsize=(6.4, 4.8)):
    """Plot a histogram of `SampleSet.prediction_probas`.

    Args:
        ss: `SampleSet` containing prediction_probas values
        bin_width: width of each bin
        n_bins: number of bins into which to bin scores
        xscale: x-axis scale
        min_bin: min value of the bin range; also sets x-axis min
        max_bin: max value of the bin range; also sets x-axis max
        yscale: y-axis scale
        ylim: tuple containing the y-axis min and max values
        title: plot title
        figsize: with and height of figure in inches

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    scores = ss.prediction_probas.values.flatten()

    BINS = np.linspace(min_bin, max_bin, n_bins)
    ax.hist(scores, bins=BINS, rwidth=bin_width)

    ax.set_xscale(xscale)
    ax.set_xlim((min_bin, max_bin))
    ax.set_yscale(yscale)
    ax.set_ylim(ylim)
    ax.set_xlabel("Scores")
    ax.set_ylabel("Occurrences")
    ax.set_title(title)
    fig.tight_layout()

    return fig, ax


def _bin_df_values_and_plot(data: pd.Series, fig, ax):
    binned_labels = data.value_counts()
    binned_labels.sort_index(inplace=True)
    binned_labels.plot(kind="bar", subplots=True, fig=fig, ax=ax)


@save_or_show_plot
def plot_label_distribution(ss: SampleSet, ylim: tuple = (1, None),
                            yscale: str = "log", figsize: tuple = (12.8, 7.2),
                            title: str = "Label Distribution",
                            target_level: str = "Isotope"):
    """Plot a histogram of `SampleSet` labels.

    Args:
        ss: `SampleSet` with `sources` values
        ylim: tuple containing the y-axis min and max values
        yscale: scale of y-axis
        figsize: width and height of figure in inches
        target_level: `SampleSet.sources` column level to use on x-axis

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = ss.get_labels(target_level=target_level)
    _bin_df_values_and_plot(labels, fig, ax)

    ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    ax.set_title(title)

    return fig, ax


@save_or_show_plot
def plot_prediction_distribution(ss: SampleSet, ylim: tuple = (1, None),
                                 yscale: str = "log", figsize: tuple = (12.8, 7.2),
                                 title: str = "Prediction Distribution",
                                 target_level: str = "Isotope"):
    """Plot a histogram of `SampleSet` predictions.

    Args:
        ss: `SampleSet` with `prediction_probas` values
        ylim: tuple containing the y-axis min and max values
        yscale: scale of y-axis
        figsize: width and height of figure in inches
        target_level: `SampleSet.sources` column level to use on x-axis

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = ss.get_predictions(target_level=target_level)
    _bin_df_values_and_plot(labels, fig, ax)

    ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    ax.set_title(title)

    return fig, ax


@save_or_show_plot
def plot_label_and_prediction_distributions(ss: SampleSet, ylim: tuple = (1, None),
                                            yscale: str = "log", figsize: tuple = (12.8, 7.2),
                                            title: str = "Label and Prediction Distribution",
                                            target_level: str = "Isotope"):
    """Plot a histogram of number of ooccurences for each label and prediction.

    Args:
        ss: `SampleSet` with label and prediction information filled in
        ylim: tuple containing the y-axis min and max values
        yscale: scale of y-axis
        figsize: with and height of figure in inches
        target_level: `SampleSet.sources` column level to use on x-axis

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = ss.get_labels(target_level=target_level)
    binned_labels = labels.value_counts()
    predictions = ss.get_predictions(target_level=target_level)
    binned_predictions = predictions.value_counts()

    binned_labels_and_predictions = pd.DataFrame(
        [binned_labels, binned_predictions],
        index=["Labels", "Predictions"]).T.fillna(0.0)
    binned_labels_and_predictions.sort_index(inplace=True)

    binned_labels.plot(kind="bar", subplots=True, fig=fig, ax=ax)

    ax.set_ylim(ylim)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(target_level)
    ax.set_ylabel("Occurences")
    fig.tight_layout()

    return fig, ax


@save_or_show_plot
def plot_correlation_between_all_labels(ss: SampleSet, mean: bool = False,
                                        figsize=(6.4, 4.8), target_level: str = "Isotope"):
    """Plot a correlation matrix of each label against every other label.

    Args:
        ss: `SampleSet` object
        mean: when True, plot the mean correlation of all enumerations of seeds,
            otherwise plot the max correlation
        figsize: with and height of figure in inches
        target_level: `SampleSet.sources` column level to use in legend

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    labels = ss.get_labels(target_level=target_level)
    X = np.zeros((len(labels), len(labels)))
    for i, label1 in enumerate(labels):
        spectra1 = ss[labels == label1].spectra
        for j, label2 in enumerate(labels):
            spectra2 = ss[labels == label2].spectra
            cur_corr = spectra1.dot(spectra2.T).values
            if mean:
                X[i, j] = np.mean(cur_corr)
            else:
                X[i, j] = np.max(cur_corr)
    X = pd.DataFrame(X, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=figsize)
    ax = heatmap(X, annot=False)
    ax.set_title(f"{'Mean' if mean else 'Max'} Correlation for Seeds")
    return fig, ax


@save_or_show_plot
def plot_precision_recall(precision, recall, marker="D", lw=2, show_legend=True, fig_ax=None,
                          title="Precision VS Recall", cmap="gist_ncar",
                          label_plot_kwargs_map=None, figsize=(6.4, 4.8)):
    """Plot the multi-class or multi-label Precision-Recall curve and mark the optimal
    F1 score for each class.

    Per-class average precision (AP) and mean average precision (mAP) are annotated on the plot.

    Args:
        precision: precision dict output of utils.precision_recall_curve()
        recall: precision dict output of utils.precision_recall_curve()
        marker: marker to use to mark the optimal F1 score point
        lw: plot line width
        show_legend: whether to display a legend
        fig_ax: optional tuple of (fig, ax) to plot on, if provided decreasing precision function
        title: plot title
        cmap: colormap to choose line colors (per label) from
        label_plot_kwargs_map: optional dictionary of (label, plot kwargs) mappings
            that will override the plot kwargs for the given label
        figsize: with and height of figure in inches.

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    from riid.models.metrics import average_precision_score, harmonic_mean

    fig, ax = fig_ax if fig_ax else plt.subplots(figsize=figsize)

    labels = [label for label in recall if label != "micro"]
    micro = ["micro"] if "micro" in recall else []

    average_precision = average_precision_score(precision, recall)
    mAP = np.mean([average_precision[label] for label in labels])

    # create F-score reference lines
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate(f"f1={f_score:0.1f}", xy=(0.85, y[45] + 0.02))

    # plot each label
    for label in labels + micro:
        f1_score = harmonic_mean(recall[label], precision[label])
        optimal_f1_idx = np.argmax(f1_score)
        optimal_f1 = f1_score[optimal_f1_idx]

        plot_kwargs = dict(lw=lw, marker=marker)

        if label == "micro":
            plot_kwargs.update(
                dict(
                    color="k",
                    linestyle=":",
                    label=f"micro-average (AP:{average_precision[label]:0.2f} "
                          f"F1*:{optimal_f1:.2f})",
                )
            )
        else:
            plot_kwargs.update(
                dict(
                    label=f"{label} (AP:{average_precision[label]:0.2f} "
                          f"F1*:{optimal_f1:.2f})",
                    color=get_label_color(label, cmap=cmap)
                )
            )

        if label_plot_kwargs_map and label in label_plot_kwargs_map:
            plot_kwargs.update(label_plot_kwargs_map[label])

        ax.plot(
            recall[label],
            precision[label],
            markevery=[optimal_f1_idx],
            **plot_kwargs,
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} (mAP: {mAP:.3f})")
    if show_legend:
        ax.legend(loc="lower left", prop=dict(size=8))

    return fig, ax


@save_or_show_plot
def plot_ss_comparison(info_stats1: dict, info_stats2: dict, col_comparisons: dict,
                       target_col: str = None, title: str = None, x_label: str = None,
                       distance_precision: int = 3):
    """Create a plot for output from `SampleSet.compare_to()`.

    Args:
        info_stats1: stats for first SampleSet
        info_stats2: stats for second SampleSet
        col_comparisons: Jensen-Shannon distance for each info column histogram
        target_col: SampleSet.info column that will be plotted
        title: plot title
        distance_precision: number of decimals to include for distance metric value

    Returns:
        Tuple (Figure, Axes) of matplotlib objects
    """
    fig, ax = plt.subplots()

    if info_stats1[target_col]["density"]:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Count")

    if x_label:
        ax.set_xlabel(x_label)
        xlbl = x_label
    else:
        ax.set_xlabel(target_col)
        xlbl = target_col

    dist_value = col_comparisons[target_col]
    if title:
        ax.set_title(f"{title}\nJ-S Distance: {round(dist_value, distance_precision)}")
    else:
        ax.set_title(f"Histogram of {xlbl} Occurrences"
                     f"\nJ-S Distance: {round(dist_value, distance_precision)}")

    stats1 = info_stats1[target_col]
    bin_width = stats1["bins"][1] - stats1["bins"][0]
    ax.bar(stats1["bins"][:-1], stats1["hist"], label="hist. 1", width=bin_width)

    stats2 = info_stats2[target_col]
    bin_width = stats2["bins"][1] - stats2["bins"][0]
    ax.bar(stats2["bins"][:-1], stats2["hist"], label="hist. 2", width=bin_width)

    ax.legend()

    return fig, ax


def get_label_color(label, cmap="gist_ncar", hashfunc=hashlib.md5) -> Tuple:
    """Choose a random color via label hash.

    Ensures the same color is always chosen for a label.

    Args:
        label: string to hash
        cmap: Matplotlib colormap
        hashfunc: hashing function

    Returns:
        Tuple of RGBA values
    """
    colormap = plt.get_cmap(cmap)
    hash_val = int(hashfunc(str(label).encode()).hexdigest(), 16)

    return colormap(hash_val % colormap.N)


class EmptyPredictionsArrayError(Exception):
    """`SampleSet.get_predictions()` returned an empty list."""
    pass
