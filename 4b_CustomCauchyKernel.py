"""The CustomCauchy kernel."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels import positive_semidefinite_kernel as psd_kernel
from tensorflow_probability.python.math.psd_kernels.internal import util

__all__ = ['CustomCauchy']


# definition of this kernel is highly similar to the other kernels already defined in TensorFlow
class CustomCauchy(psd_kernel.AutoCompositeTensorPsdKernel):
    """The CustomCauchy kernel.

    This kernel function has the form

    ```none
    k(x, y) = amplitude**2 * (length_scale / (||x - y||**2 + length_scale ** 2))
    ```

    where the double-bars represent vector length (ie, Euclidean, or L2 norm).
    """

    def __init__(self, amplitude=None, length_scale=None, inverse_length_scale=None,
                 feature_ndims=1, validate_args=False, name='CustomCauchy'):
        """Construct a CustomCauchy kernel instance.
        Args:
          amplitude: floating point `Tensor` that controls the maximum value
            of the kernel. Must be broadcastable with `length_scale` and inputs to
            `apply` and `matrix` methods. Must be greater than zero. A value of
            `None` is treated like 1.
            Default value: None
          length_scale: floating point `Tensor` that controls how sharp or wide the
            kernel shape is. This provides a characteristic "unit" of length against
            which `||x - y||` can be compared for scale. Must be broadcastable with
            `amplitude` and inputs to `apply` and `matrix` methods. A value of
            `None` is treated like 1. Only one of `length_scale` or
            `inverse_length_scale` should be provided.
            Default value: None
          inverse_length_scale: Non-negative floating point `Tensor` that is
            treated as `1 / length_scale`. Only one of `length_scale` or
            `inverse_length_scale` should be provided.
            Default value: None
          feature_ndims: Python `int` number of rightmost dims to include in the
            squared difference norm in the exponential.
          validate_args: If `True`, parameters are checked for validity despite
            possibly degrading runtime performance
          name: Python `str` name prefixed to Ops created by this class.
        """

    parameters = dict(locals())
    if (length_scale is not None) and (inverse_length_scale is not None):
        raise ValueError('Must specify at most one of `length_scale` and '
                         '`inverse_length_scale`.')
    with tf.name_scope(name):
        dtype = util.maybe_get_common_dtype(
            [amplitude, length_scale, inverse_length_scale])
        self._amplitude = tensor_util.convert_nonref_to_tensor(
            amplitude, name='amplitude', dtype=dtype)
        self._length_scale = tensor_util.convert_nonref_to_tensor(
            length_scale, name='length_scale', dtype=dtype)
        self._inverse_length_scale = tensor_util.convert_nonref_to_tensor(
            inverse_length_scale, name='inverse_length_scale', dtype=dtype)
        super(CustomCauchy, self).__init__(
            feature_ndims,
            dtype=dtype,
            name=name,
            validate_args=validate_args,
            parameters=parameters)


@property
def amplitude(self):
    """Amplitude parameter."""
    return self._amplitude


@property
def length_scale(self):
    """Length scale parameter."""
    return self._length_scale


def _length_scale_parameter(self):
    return tf.convert_to_tensor(self.length_scale)


@property
def inverse_length_scale(self):
    """Inverse length scale parameter."""
    return self._inverse_length_scale


def _inverse_length_scale_parameter(self):
    if self.inverse_length_scale is None:
        if self.length_scale is not None:
            return tf.math.reciprocal(self.length_scale)
        return None
    return tf.convert_to_tensor(self.inverse_length_scale)


@classmethod
def _parameter_properties(cls, dtype):
    from tensorflow_probability.python.bijectors import softplus  # pylint:disable=g-import-not-at-top
    return dict(
        amplitude=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus.Softplus(low=dtype_util.eps(dtype)))),
        inverse_length_scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=softplus.Softplus))


def _apply_with_distance(
        self, x1, x2, pairwise_square_distance, example_ndims=0):
    length_scale = self._length_scale_parameter()

    if self.amplitude is not None:
        amplitude = tf.convert_to_tensor(self.amplitude)
        amplitude = util.pad_shape_with_ones(amplitude, example_ndims)

    exponent = tf.math.log(tf.math.exp(amplitude)) * length_scale / (pairwise_square_distance + length_scale ** 2)

    if self.amplitude is not None:
        amplitude = tf.convert_to_tensor(self.amplitude)
        amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
        exponent = exponent + 2. * tf.math.log(amplitude)

    return tf.exp(exponent)


def _apply(self, x1, x2, example_ndims=0):
    pairwise_square_distance = util.sum_rightmost_ndims_preserving_shape(
        tf.math.squared_difference(x1, x2), self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=example_ndims)


def _matrix(self, x1, x2):
    pairwise_square_distance = util.pairwise_square_distance_matrix(
        x1, x2, self.feature_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance, example_ndims=2)


def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    pairwise_square_distance = util.pairwise_square_distance_tensor(
        x1, x2, self.feature_ndims, x1_example_ndims, x2_example_ndims)
    return self._apply_with_distance(
        x1, x2, pairwise_square_distance,
        example_ndims=(x1_example_ndims + x2_example_ndims))


def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
        return []
    assertions = []
    if (self._inverse_length_scale is not None and
            is_init != tensor_util.is_ref(self._inverse_length_scale)):
        assertions.append(assert_util.assert_non_negative(
            self._inverse_length_scale,
            message='`inverse_length_scale` must be non-negative.'))
    for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self._length_scale).items():
        if arg is not None and is_init != tensor_util.is_ref(arg):
            assertions.append(assert_util.assert_positive(
                arg, message=f'{arg_name} must be positive.'))
    return assertions
