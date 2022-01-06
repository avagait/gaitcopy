import tensorflow as tf


class ModelSaver(tf.keras.callbacks.Callback):
	def __init__(self, model_object, **kwargs):
		super().__init__()
		self.model_object = model_object

		self.every = 1
		for key in kwargs:
			if key == "every":
				self.every = kwargs[key]

	def on_epoch_end(self, epoch, logs=None):
		if (self.every == 1) or (epoch % self.every == 0):
			self.model_object.save(epoch)
