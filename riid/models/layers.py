# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains custom Keras layers."""
import tensorflow as tf
from keras.api.layers import Layer


class ReduceSumLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis):
        return tf.reduce_sum(x, axis=axis)


class ReduceMaxLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.reduce_max(x)


class DivideLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.divide(x[0], x[1])


class ExpandDimsLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis):
        return tf.expand_dims(x, axis=axis)


class ClipByValueLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, clip_value_min, clip_value_max):
        return tf.clip_by_value(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)


class PoissonLogProbabilityLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        exp, value = x
        log_probas = tf.math.xlogy(value, exp) - exp - tf.math.lgamma(value + 1)
        return log_probas


class SeedLayer(Layer):
    def __init__(self, seeds, **kwargs):
        super(SeedLayer, self).__init__(**kwargs)
        self.seeds = tf.convert_to_tensor(seeds)

    def get_config(self):
        config = super().get_config()
        config.update({
            "seeds": self.seeds.numpy().tolist(),
        })
        return config

    def call(self, inputs):
        return self.seeds


class L1NormLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        sums = tf.reduce_sum(inputs, axis=-1)
        l1_norm = inputs / tf.reshape(sums, (-1, 1))
        return l1_norm
