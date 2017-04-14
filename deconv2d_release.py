from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from prettytensor.pretty_tensor_image_methods import _kernel, _stride
import tensorflow as tf

from prettytensor import layers, parameters
from prettytensor import pretty_tensor_class as prettytensor
from prettytensor import pretty_tensor_normalization_methods
from prettytensor.pretty_tensor_class import PAD_SAME
from prettytensor.pretty_tensor_class import PROVIDED
# pylint: disable=redefined-outer-name,invalid-name
@prettytensor.Register(
    assign_defaults=('activation_fn', 'l2loss', 'batch_normalize',
                     'parameter_modifier', 'phase'))
class deconv2d(prettytensor.VarStoreMethod):

  def __call__(self,
               input_layer,
               kernel,
               depth,
               output_shape,
               activation_fn=None,
               stride=(1, 1),
               l2loss=None,
               weights=None,
               bias=tf.zeros_initializer(),
               edges=PAD_SAME,
               batch_normalize=False,
               phase=prettytensor.Phase.train,
               parameter_modifier=parameters.identity,
               name=PROVIDED):
    """Adds a convolution to the stack of operations.

    `kernel` is the patch that will be pooled and it describes the pooling
    along each of the 4 dimensions.  The stride is how big to take each step.

    * scalar (e.g. 3): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * singleton list (e.g. [3]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 3, 1]`).
    * list of length 2 (e.g. [3, 2]): Square pooling on the image
        (`[b, c, r, d] = [1, 3, 2, 1]`).

    Args:
      input_layer: The chainable object, supplied.
      kernel: The size of the patch for the pool, either an int or a length 1 or
        2 sequence (if length 1 or int, it is expanded).
      depth: The depth of the new Tensor.
      activation_fn: A tuple of (activation_function, extra_parameters). Any
        function that takes a tensor as its first argument can be used. More
        common functions will have summaries added (e.g. relu).
      stride: The strides as a length 1, 2 or 4 sequence or an integer. If an
        int, length 1 or 2, the stride in the first and last dimensions are 1.
      l2loss: Set to a value greater than 0 to use L2 regularization to decay
        the weights.
      weights:  An initializer for weights or a Tensor. If not specified,
        uses He's initialization.
      bias: An initializer for the bias or a Tensor. No bias if set to None.
      edges: Either SAME to use 0s for the out of bounds area or VALID to shrink
        the output size and only uses valid input pixels.
      batch_normalize: Supply a BatchNormalizationArguments to set the
        parameters for batch normalization.
      phase: The phase of graph construction.  See `pt.Phase`.
      parameter_modifier: A function to modify parameters that is applied after
        creation and before use.
      name: The name for this operation is also used to create/find the
        parameter variables.
    Returns:
      Handle to the generated layer.
    Raises:
      ValueError: If input_layer is not a rank 4 tensor or the  depth of the
        input (4th dim) is not known.
    """
    if input_layer.get_shape().ndims != 4:
      raise ValueError('conv2d requires a rank 4 Tensor with a known depth %s' %
                       input_layer.get_shape())
    if input_layer.shape[3] is None:
      raise ValueError('Input depth must be known')
    kernel = _kernel(kernel)
    stride = _stride(stride)
    size = [kernel[0], kernel[1], depth, input_layer.shape[3]]
    #size = [kernel[0], kernel[1], depth, input_layer.shape[3]] key change

    books = input_layer.bookkeeper
    if weights is None:
      patch_size = size[0] * size[1]
      weights = layers.he_init(size[2] * patch_size, size[3] * patch_size,
                               activation_fn)

    dtype = input_layer.tensor.dtype
    params = parameter_modifier(
        'weights',
        self.variable('weights', size, weights, dt=dtype),
        phase)
    output_shape[0]=input_layer.shape[0]
    y = tf.nn.conv2d_transpose(input_layer, params, output_shape, stride, padding=edges)
    #output_shape[0]=input_layer.shape[0] key change
    #y = tf.nn.conv2d_transpose(input_layer, params, output_shape, stride, padding=edges) key change
    layers.add_l2loss(books, params, l2loss)
    if bias is not None:
      y += parameter_modifier('bias',
                              self.variable('bias', [size[-2]],
                                            bias,
                                            dt=dtype),
                              phase)
      # y += parameter_modifier('bias',
      #                         self.variable('bias', [size[-2]],
      #                                       bias,
      #                                       dt=dtype),
      #                         phase) size parameter location changed due to deconv2d
    books.add_scalar_summary(
        tf.reduce_mean(layers.spatial_slice_zeros(y)),
        '%s/zeros_spatial' % y.op.name)
    y = pretty_tensor_normalization_methods.batch_normalize_with_arguments(
        y, batch_normalize)
    if activation_fn is not None:
      if not isinstance(activation_fn, collections.Sequence):
        activation_fn = (activation_fn,)
      y = layers.apply_activation(books,
                                  y,
                                  activation_fn[0],
                                  activation_args=activation_fn[1:])
    books.add_histogram_summary(y, '%s/activations' % y.op.name)
    return input_layer.with_tensor(y, parameters=self.vars)