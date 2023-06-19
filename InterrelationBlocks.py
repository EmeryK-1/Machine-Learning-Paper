import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import regularizers


class CompanionMatrixBlock(Layer):
    def __init__(self,
                 num_time_series,
                 horizon,
                 back_horizon,
                 initializer="glorot_uniform",
                 trainable_backcast=True,
                 **kwargs):
        super().__init__(**kwargs)

        self._num_time_series = num_time_series
        self._back_horizon = back_horizon
        self._horizon = horizon
        self._companion_back_horizon = self.add_weight(shape=(num_time_series, num_time_series * back_horizon),
                                                       trainable=trainable_backcast,
                                                       initializer=initializer,
                                                       name='FC_back_horizon_companion')

        self._companion_horizon = self.add_weight(shape=(num_time_series, num_time_series * back_horizon),
                                                  trainable=True,
                                                  initializer=initializer,
                                                  name='FC_horizon_companion')

    def call(self, inputs):
        flipped_inputs = tf.reverse(inputs, axis=[1])
        flattened = tf.reshape(flipped_inputs, (-1, self._back_horizon * self._num_time_series))[:, :self._num_time_series * self._back_horizon]
        y_back_horizon = tf.linalg.matvec(self._companion_back_horizon, flattened)
        y_horizon = tf.linalg.matvec(self._companion_horizon, flattened)
        # Reshape to be (Batch size, horizon, num_time_series)
        y_back_horizon = tf.reshape(y_back_horizon, (-1, self._horizon, self._num_time_series))
        y_horizon = tf.reshape(y_horizon, (-1, self._horizon, self._num_time_series))
        return y_horizon, y_back_horizon


class CompanionMatrixBlock(Layer):
    def __init__(self,
                 num_time_series,
                 horizon,
                 back_horizon,
                 lags,
                 initializer="glorot_uniform",
                 trainable_backcast=True,
                 regularizer=regularizers.l1(0.01),
                 **kwargs):
        super().__init__(**kwargs)

        self._num_time_series = num_time_series
        self._back_horizon = back_horizon
        self._horizon = horizon
        self._lags = lags

        self._companion_horizon = self.add_weight(shape=(num_time_series, num_time_series * lags),
                                                  trainable=True,
                                                  initializer=initializer,
                                                  regularizer=regularizer,
                                                  name='FC_horizon_companion')

        self._companion_back_horizon = self.add_weight(shape=(num_time_series, num_time_series * lags),
                                                       trainable=trainable_backcast,
                                                       initializer=initializer,
                                                       name='FC_back_horizon_companion')
        self._past_input = None

    def call(self, inputs, training=False):

        if not training:
            self._past_input = inputs
        companion_horizon = tf.concat([
            self._companion_horizon,
            tf.pad(tf.eye(self._num_time_series * (self._lags - 1)), [[0, 0], [0, self._num_time_series]])
        ], axis=0)
        companion_back_horizon = tf.concat([
            self._companion_back_horizon,
            tf.pad(tf.eye(self._num_time_series * (self._lags - 1)), [[0, 0], [0, self._num_time_series]])
        ], axis=0)
        # Invert input
        inputs = tf.reverse(inputs, axis=[1])
        # Inputs shape is (batch_size, back_horizon, num_time_series)
        flattened = tf.reshape(inputs, (-1, self._back_horizon * self._num_time_series))
        y_horizon = [tf.transpose(companion_horizon @ tf.transpose(flattened[:, :self._num_time_series * self._lags]))]

        for _ in range(self._horizon - 1):
            y_horizon.append(tf.transpose(companion_horizon @ tf.transpose(y_horizon[-1])))
        y_horizon = tf.stack(y_horizon, axis=1)
        # Take the first num_time_series elements of the forecast (2nd dimension)
        y_horizon = y_horizon[:, :, :self._num_time_series]
        # Add lags rows of 0s to the top of inputs
        flattened = tf.pad(flattened, [[0, 0], [0, self._num_time_series * self._lags]])
        y_back_horizon = []
        for i in range(self._back_horizon):
            y_back_horizon.append(tf.transpose(companion_back_horizon @ tf.transpose(flattened[:, i * self._num_time_series:(i + self._lags) * self._num_time_series])))
        y_back_horizon = tf.stack(y_back_horizon, axis=1)
        y_back_horizon = y_back_horizon[:, :, :self._num_time_series]
        return y_horizon, y_back_horizon
