# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.

# This code is based on Tensorflow-Addons. THE ORIGINAL CODE HAS BEEN MODIFIED.
# https://www.tensorflow.org/addons/

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains sparsemax-related functions."""

from typing import Optional

import tensorflow as tf
from typeguard import typechecked


def sparsemax(logits, axis: int = -1) -> tf.Tensor:
    r"""Sparsemax activation function.

    For each batch \( i \), and class \( j \),
    compute sparsemax activation function:

    $$
    \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
    $$

    See
    [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
    ](https://arxiv.org/abs/1602.02068).

    Usage:

    >>> x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> tfa.activations.sparsemax(x)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 1.],
           [0., 0., 1.]], dtype=float32)>

    Args:
        logits: A `Tensor`.
        axis: `int`, axis along which the sparsemax operation is applied.

    Returns:
        A `Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.

    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


def sparsemax_loss(logits, sparsemax, labels, name: Optional[str] = None) -> tf.Tensor:
    r"""Sparsemax loss function ([1]).

    Computes the generalized multi-label classification loss for the sparsemax
    function. The implementation is a reformulation of the original loss
    function such that it uses the sparsemax probability output instead of the
    internal \( \tau \) variable. However, the output is identical to the original
    loss function.

    [1]: https://arxiv.org/abs/1602.02068

    Args:
      logits: A `Tensor`. Must be one of the following types: `float32`,
        `float64`.
      sparsemax: A `Tensor`. Must have the same type as `logits`.
      labels: A `Tensor`. Must have the same type as `logits`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `logits`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")
    sparsemax = tf.convert_to_tensor(sparsemax, name="sparsemax")
    labels = tf.convert_to_tensor(labels, name="labels")

    # In the paper, they call the logits z.
    # A constant can be substracted from logits to make the algorithm
    # more numerically stable in theory. However, there are really no major
    # source numerical instability in this algorithm.
    z = logits

    # sum over support
    # Use a conditional where instead of a multiplication to support z = -inf.
    # If z = -inf, and there is no support (sparsemax = 0), a multiplication
    # would cause 0 * -inf = nan, which is not correct in this case.
    sum_s = tf.where(
        tf.math.logical_or(sparsemax > 0, tf.math.is_nan(sparsemax)),
        sparsemax * (z - 0.5 * sparsemax),
        tf.zeros_like(sparsemax),
    )

    # - z_k + ||q||^2
    q_part = labels * (0.5 * labels - z)
    # Fix the case where labels = 0 and z = -inf, where q_part would
    # otherwise be 0 * -inf = nan. But since the lables = 0, no cost for
    # z = -inf should be consideredself.
    # The code below also coveres the case where z = inf. Howeverm in this
    # caose the sparsemax will be nan, which means the sum_s will also be nan,
    # therefor this case doesn't need addtional special treatment.
    q_part_safe = tf.where(
        tf.math.logical_and(tf.math.equal(labels, 0), tf.math.is_inf(z)),
        tf.zeros_like(z),
        q_part,
    )

    return tf.math.reduce_sum(sum_s + q_part_safe, axis=1)


@tf.function
@tf.keras.utils.register_keras_serializable(package="Addons")
def sparsemax_loss_from_logits(
    y_true, logits_pred
) -> tf.Tensor:
    y_pred = sparsemax(logits_pred)
    loss = sparsemax_loss(logits_pred, y_pred, y_true)
    return loss


@tf.keras.utils.register_keras_serializable(package="Addons")
class SparsemaxLoss(tf.keras.losses.Loss):
    """Sparsemax loss function.

    Computes the generalized multi-label classification loss for the sparsemax
    function.

    Because the sparsemax loss function needs both the probability output and
    the logits to compute the loss value, `from_logits` must be `True`.

    Because it computes the generalized multi-label loss, the shape of both
    `y_pred` and `y_true` must be `[batch_size, num_classes]`.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. Default
        is `True`, meaning `y_pred` is the logits.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "sparsemax_loss",
    ):
        if from_logits is not True:
            raise ValueError("from_logits must be True")

        super().__init__(name=name, reduction=reduction)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return sparsemax_loss_from_logits(y_true, y_pred)

    def get_config(self):
        config = {
            "from_logits": self.from_logits,
        }
        base_config = super().get_config()
        return {**base_config, **config}
