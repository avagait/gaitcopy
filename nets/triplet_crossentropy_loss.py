import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional

def _masked_maximum(data, mask, dim=1):
	"""Computes the axis wise maximum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the maximum.

	Returns:
	  masked_maximums: N-D `Tensor`.
		The maximized dimension is of size 1 after the operation.
	"""
	axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
	masked_maximums = (
		tf.math.reduce_max(
			tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
		)
		+ axis_minimums
	)
	return masked_maximums


def _masked_minimum(data, mask, dim=1):
	"""Computes the axis wise minimum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the minimum.

	Returns:
	  masked_minimums: N-D `Tensor`.
		The minimized dimension is of size 1 after the operation.
	"""
	axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
	masked_minimums = (
		tf.math.reduce_min(
			tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
		)
		+ axis_maximums
	)
	return masked_minimums


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def triplet_semihard_crossentropy_loss(
	y_true: TensorLike, y_pred: TensorLike, margin: FloatTensorLike = 1.0
) -> tf.Tensor:
	"""Computes the triplet loss with semi-hard negative mining.

	Args:
	  y_true: 1-D integer `Tensor` with shape [batch_size] of
		multiclass integer labels.
	  y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should
		be l2 normalized.
	  margin: Float, margin term in the loss definition.

	Returns:
	  triplet_loss: float scalar with dtype of y_pred.
	"""
	labels, embeddings = y_true, y_pred

	convert_to_float32 = (
		embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
	)
	precise_embeddings = (
		tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
	)

	# Reshape label tensor to [batch_size, 1].
	lshape = tf.shape(labels)
	labels = tf.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = metric_learning.pairwise_distance(precise_embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = tf.math.equal(labels, tf.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = tf.math.logical_not(adjacency)

	batch_size = tf.size(labels)

	# Compute the mask.
	pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
	mask = tf.math.logical_and(
		tf.tile(adjacency_not, [batch_size, 1]),
		tf.math.greater(
			pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
		),
	)
	mask_final = tf.reshape(
		tf.math.greater(
			tf.math.reduce_sum(
				tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
			),
			0.0,
		),
		[batch_size, batch_size],
	)
	mask_final = tf.transpose(mask_final)

	adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
	mask = tf.cast(mask, dtype=tf.dtypes.float32)

	# negatives_outside: smallest D_an where D_an > D_ap.
	negatives_outside = tf.reshape(
		_masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
	)
	negatives_outside = tf.transpose(negatives_outside)

	# negatives_inside: largest D_an.
	negatives_inside = tf.tile(
		_masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
	)
	semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

#	loss_mat = tf.math.add(margin, pdist_matrix - semi_hard_negatives)
	loss_mat = pdist_matrix - semi_hard_negatives

	mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
		tf.ones([batch_size])
	)

	# In lifted-struct, the authors multiply 0.5 for upper triangular
	#   in semihard, they take all positive pairs except the diagonal.
	num_positives = tf.math.reduce_sum(mask_positives)

#	distances = tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
#	distances = tf.math.maximum(loss_mat, 0.0)
	distances = tf.clip_by_value(loss_mat, -100.0, 100.0)

#	triplet_loss = tf.math.truediv(
#		tf.math.reduce_sum(
#			tf.math.log(tf.constant(1.0), tf.constant(1.0) + tf.math.exp(distances))
#		),
#		num_positives,
#	)
	print("dist shape", distances.shape)
	triplet_loss = tf.math.truediv(
		tf.math.reduce_sum(
			tf.math.multiply(tf.nn.softplus(distances), mask_positives)
		),
		num_positives,
	)

	if convert_to_float32:
		return tf.cast(triplet_loss, embeddings.dtype)
	else:
		return triplet_loss

class TripletSemiHardCrossentropyLoss(tf.keras.losses.Loss):
	"""Computes the triplet loss with semi-hard negative mining.
	The loss encourages the positive distances (between a pair of embeddings
	with the same labels) to be smaller than the minimum negative distance
	among which are at least greater than the positive distance plus the
	margin constant (called semi-hard negative) in the mini-batch.
	If no such negative exists, uses the largest negative distance instead.
	See: https://arxiv.org/abs/1503.03832.
	We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
	[batch_size] of multi-class integer labels. And embeddings `y_pred` must be
	2-D float `Tensor` of l2 normalized embedding vectors.
	Args:
	  margin: Float, margin term in the loss definition. Default value is 1.0.
	  name: Optional name for the op.
	"""

	@typechecked
	def __init__(
		self, margin: FloatTensorLike = 1.0, name: Optional[str] = None, **kwargs
	):
		super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
		self.margin = margin

	def call(self, y_true, y_pred):
		return triplet_semihard_crossentropy_loss(y_true, y_pred, self.margin)

	def get_config(self):
		config = {
			"margin": self.margin,
		}
		base_config = super().get_config()
		return {**base_config, **config}
