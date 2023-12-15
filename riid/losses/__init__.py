# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains custom loss functions."""
import numpy as np
import tensorflow as tf
from keras import backend as K


def negative_log_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate negative log F1 score.

    Args:
        y_true: list of ground truth
        y_pred: list of predictions to compare against the ground truth

    Returns:
        Custom loss score on a log scale
    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(K.log(true_positive))


def negative_f1(y_true, y_pred):
    """Calculate negative F1 score.

    Args:
        y_true: list of ground truth
        y_pred: list of predictions to compare against the ground truth

    Returns:
        Custom loss score
    """
    diff = y_true - y_pred
    negs = K.clip(diff, -1.0, 0.0)
    false_positive = -K.sum(negs, axis=-1)
    true_positive = 1.0 - false_positive
    lower_clip = 1e-20
    true_positive = K.clip(true_positive, lower_clip, 1.0)

    return -K.mean(true_positive)


def build_keras_semisupervised_loss_func(supervised_loss_func,
                                         unsupervised_loss_func,
                                         dictionary, beta,
                                         activation, n_labels,
                                         normalize: bool = False,
                                         normalize_scaler: float = 1.0,
                                         normalize_func=tf.math.tanh):
    def _semisupervised_loss_func(data, y_pred):
        """
        Args:
            data: Contains true labels and input features (spectra)
            y_pred: Model output (unactivated logits)
        """
        y_true = data[:, :n_labels]
        spectra = data[:, n_labels:]
        logits = y_pred
        lpes = activation(y_pred)

        sup_losses = supervised_loss_func(y_true, logits)
        unsup_losses = reconstruction_error(spectra, lpes, dictionary,
                                            unsupervised_loss_func)
        if normalize:
            sup_losses = normalize_func(normalize_scaler * sup_losses)

        semisup_losses = (1 - beta) * sup_losses + beta * unsup_losses

        return semisup_losses

    return _semisupervised_loss_func


def sse_diff(spectra, reconstructed_spectra):
    """Compute the sum of squares error.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: spectral samples, assumed to be in counts
        reconstructed_spectra: reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    diff = spectra - scaled_reconstructed_spectra
    norm_diff = tf.norm(diff, axis=-1)
    squared_norm_diff = tf.square(norm_diff)
    return squared_norm_diff


def poisson_nll_diff(spectra, reconstructed_spectra, eps=1e-8):
    """Compute the Poisson Negative Log-Likelihood.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: spectral samples, assumed to be in counts
        reconstructed_spectra: reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=-1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )
    log_reconstructed_spectra = tf.math.log(scaled_reconstructed_spectra + eps)
    diff = tf.nn.log_poisson_loss(
        spectra,
        log_reconstructed_spectra,
        compute_full_loss=True
    )
    diff = tf.reduce_sum(diff, axis=-1)

    return diff


def normal_nll_diff(spectra, reconstructed_spectra, eps=1e-8):
    """Compute the Normal Negative Log-Likelihood.

    TODO: refactor to assume spectral inputs are in the same form

    Args:
        spectra: spectral samples, assumed to be in counts
        reconstructed_spectra: reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=-1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    var = tf.clip_by_value(spectra, clip_value_min=1, clip_value_max=np.inf)

    sigma_term = tf.math.log(2 * np.pi * var)
    mu_term = tf.math.divide(tf.math.square(scaled_reconstructed_spectra - spectra), var)
    diff = sigma_term + mu_term
    diff = 0.5 * tf.reduce_sum(diff, axis=-1)

    return diff


def weighted_sse_diff(spectra, reconstructed_spectra):
    """Compute the Normal Negative Log-Likelihood under constant variance
    (this reduces to the SSE, just on a different scale).

    Args:
        spectra: spectral samples, assumed to be in counts
        reconstructed_spectra: reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    sample_variance = tf.sqrt(tf.math.reduce_variance(spectra, axis=1))

    sigma_term = tf.math.log(2 * np.pi * sample_variance)

    mu_term = tf.math.divide(
        tf.math.square(scaled_reconstructed_spectra - spectra),
        tf.reshape(sample_variance, (-1, 1))
    )
    diff = 0.5 * (sigma_term + tf.reduce_sum(mu_term, axis=-1))

    return diff


def reconstruction_error(spectra, lpes, dictionary, diff_func):
    reconstructed_spectra = tf.matmul(lpes, dictionary)
    reconstruction_errors = diff_func(spectra, reconstructed_spectra)
    return reconstruction_errors


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def jensen_shannon_divergence(p, q):
    p_sum = tf.reduce_sum(p, axis=-1)
    p_norm = tf.divide(
        p,
        tf.reshape(p_sum, (-1, 1))
    )

    q_sum = tf.reduce_sum(q, axis=-1)
    q_norm = tf.divide(
        q,
        tf.reshape(q_sum, (-1, 1))
    )

    kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    m = (p_norm + q_norm) / 2
    jsd = (kld(p_norm, m) + kld(q_norm, m)) / 2
    return jsd


def jensen_shannon_distance(p, q):
    divergence = jensen_shannon_divergence(p, q)
    return tf.math.sqrt(divergence)


def chi_squared_diff(spectra, reconstructed_spectra):
    """Compute the Chi-Squared test.

    Args:
        spectra: spectral samples, assumed to be in counts
        reconstructed_spectra: reconstructed spectra created using a
            dictionary with label proportion estimates
    """
    total_counts = tf.reduce_sum(spectra, axis=1)
    scaled_reconstructed_spectra = tf.multiply(
        reconstructed_spectra,
        tf.reshape(total_counts, (-1, 1))
    )

    diff = tf.math.subtract(spectra, scaled_reconstructed_spectra)
    squared_diff = tf.math.square(diff)
    variances = tf.clip_by_value(spectra, 1, np.inf)
    chi_squared = tf.math.divide(squared_diff, variances)
    return tf.reduce_sum(chi_squared, axis=-1)
