from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export
import tensorflow as tf

class BaseDenseAttention(Layer):

  def __init__(self, causal=False, dropout=0.0, **kwargs):
    super(BaseDenseAttention, self).__init__(**kwargs)
    self.causal = causal
    self.dropout = dropout
    self.supports_masking = True

  def _calculate_scores(self, query, key):
    return NotImplementedError

  def _apply_scores(self, scores, value, scores_mask=None, training=None):
  
    if scores_mask is not None:
      padding_mask = math_ops.logical_not(scores_mask)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
    if training is None:
      training = K.learning_phase()
    weights = nn.softmax(scores)

    def dropped_weights():
      return nn.dropout(weights, rate=self.dropout)

    weights = tf_utils.smart_cond(
        training,
        dropped_weights,
        lambda: array_ops.identity(weights))
    return weights, math_ops.matmul(weights, value)

  # TODO(b/125916026): Consider exposing a __call__ method with named args.
  def call(self, inputs, mask=None, training=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    scores = self._calculate_scores(query=q, key=k)
    if v_mask is not None:
      # Mask of shape [batch_size, 1, Tv].
      v_mask = array_ops.expand_dims(v_mask, axis=-2)
    if self.causal:
      # Creates a lower triangular mask, so position i cannot attend to
      # positions j>i. This prevents the flow of information from the future
      # into the past.
      scores_shape = array_ops.shape(scores)
      # causal_mask_shape = [1, Tq, Tv].
      causal_mask_shape = array_ops.concat(
          [array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]],
          axis=0)
      causal_mask = _lower_triangular_mask(causal_mask_shape)
    else:
      causal_mask = None
    scores_mask = _merge_masks(v_mask, causal_mask)
    weights, result = self._apply_scores(
        scores=scores, value=v, scores_mask=scores_mask, training=training)
    if q_mask is not None:
      # Mask of shape [batch_size, Tq, 1].
      q_mask = array_ops.expand_dims(q_mask, axis=-1)
      result *= math_ops.cast(q_mask, dtype=result.dtype)
    return weights, result

  def compute_mask(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    if mask:
      q_mask = mask[0]
      if q_mask is None:
        return None
      return ops.convert_to_tensor_v2(q_mask)
    return None

  def _validate_call_args(self, inputs, mask):
    """Validates arguments of the call method."""
    class_name = self.__class__.__name__
    if not isinstance(inputs, list):
      raise ValueError(
          '{} layer must be called on a list of inputs, namely [query, value] '
          'or [query, value, key].'.format(class_name))
    if len(inputs) < 2 or len(inputs) > 3:
      raise ValueError(
          '{} layer accepts inputs list of length 2 or 3, '
          'namely [query, value] or [query, value, key]. '
          'Given length: {}'.format(class_name, len(inputs)))
    if mask:
      if not isinstance(mask, list):
        raise ValueError(
            '{} layer mask must be a list, '
            'namely [query_mask, value_mask].'.format(class_name))
      if not len(mask) in {2,3}:
        raise ValueError(
            '{} layer mask must be a list of length 2 or 3, namely [query_mask, '
            'value_mask] or .... Given length: {}'.format(class_name, len(mask)))

  def get_config(self):
    config = {
        'causal': self.causal,
        'dropout': self.dropout,
    }
    base_config = super(BaseDenseAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Attention')
class Attention(BaseDenseAttention):

  def __init__(self, use_scale=False, **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.use_scale = use_scale

  def build(self, input_shape):
    """Creates scale variable if use_scale==True."""
    if self.use_scale:
      self.scale = self.add_weight(
          name='scale',
          shape=(),
          initializer=init_ops.ones_initializer(),
          dtype=self.dtype,
          trainable=True)
    else:
      self.scale = None
    super(Attention, self).build(input_shape)

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a query-key dot product.
    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    scores = math_ops.matmul(query, key, transpose_b=True)
    if self.scale is not None:
      scores *= self.scale
    return scores

  def get_config(self):
    config = {'use_scale': self.use_scale}
    base_config = super(Attention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.AdditiveAttention')
class AdditiveAttention(BaseDenseAttention):
  
  def __init__(self, use_scale=True, **kwargs):
    super(AdditiveAttention, self).__init__(**kwargs)
    self.use_scale = use_scale

  def build(self, input_shape):
    v_shape = tensor_shape.TensorShape(input_shape[1])
    dim = v_shape[-1]
    if isinstance(dim, tensor_shape.Dimension):
      dim = dim.value
    if self.use_scale:
      self.scale = self.add_weight(
          name='scale',
          shape=[dim],
          initializer=init_ops.glorot_uniform_initializer(),
          dtype=self.dtype,
          trainable=True)
    else:
      self.scale = None
    super(AdditiveAttention, self).build(input_shape)

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a nonlinear sum of query and key.
    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    # Reshape tensors to enable broadcasting.
    # Reshape into [batch_size, Tq, 1, dim].
    q_reshaped = array_ops.expand_dims(query, axis=-2)
    # Reshape into [batch_size, 1, Tv, dim].
    k_reshaped = array_ops.expand_dims(key, axis=-3)
    if self.use_scale:
      scale = self.scale
    else:
      scale = 1.
    return math_ops.reduce_sum(
        scale * math_ops.tanh(q_reshaped + k_reshaped), axis=-1)

  def get_config(self):
    config = {'use_scale': self.use_scale}
    base_config = super(AdditiveAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _lower_triangular_mask(shape):
  """Creates a lower-triangular boolean mask over the last 2 dimensions."""
  row_index = math_ops.cumsum(
      array_ops.ones(shape=shape, dtype=dtypes.int32), axis=-2)
  col_index = math_ops.cumsum(
      array_ops.ones(shape=shape, dtype=dtypes.int32), axis=-1)
  return math_ops.greater_equal(row_index, col_index)


def _merge_masks(x, y):
  if x is None:
    return y
  if y is None:
    return x
  return math_ops.logical_and(x, y)
  
class MultiheadAttention(tf.Module):
    def __init__(self, num_heads, d_model, attention_type='additive', name=None):
        super(MultiheadAttention, self).__init__(name=name)
        self.heads = []
        self.dense_layers = [[tf.keras.layers.Dense(d_model, name='dense_'+str(i)+str(j)) for j in range(3)] for i in range(num_heads)]
        self.final_dense = tf.keras.layers.Dense(d_model, name='dense_final')
        with self.name_scope:
            for idx in range(num_heads):
                if attention_type=='additive':
                    self.heads.append(AdditiveAttention(name='Additive_Attention_Head_'+str(idx+1)))
                elif attention_type=='scaled_dot_product':
                    self.heads.append(Attention(name='Attention_Head_'+str(idx+1)))
    @tf.Module.with_name_scope
    def __call__(self, x):
        outputs = []
        for head, curr_dense in zip(self.heads, self.dense_layers):
            curr_q = x[0]
            curr_v = x[1]
            curr_k = x[1]
            projected_q = curr_dense[0](curr_q)
            projected_v = curr_dense[1](curr_v)
            projected_k = curr_dense[2](curr_k)
            head_output = head([projected_q, projected_v, projected_k])[1]
            outputs.append(head_output)
        concatenation = tf.concat(outputs, -1)
        return self.final_dense(concatenation)
        