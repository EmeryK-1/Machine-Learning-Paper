import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from UniBlocks import SimpleTrendBlock, SimpleSeasonalityBlock


class MultiBlock(Layer):
    def __init__(self,
                 horizon,
                 back_horizon,
                 blocks,
                 **kwargs):
        super().__init__(**kwargs)
        self._horizon = horizon
        self._back_horizon = back_horizon
        self._blocks = blocks # Each block is a for a separate time series

    def call(self, inputs):
        y_horizons = []
        y_back_horizons = []
        for i in range(len(self._blocks)):  # For each block, forward pass the data for it's given time series
            y_horizon, y_back_horizon = self._blocks[i](inputs[:, :, i])  # shape: (Batch_size, horizon),
            y_horizons.append(y_horizon)
            y_back_horizons.append(y_back_horizon)

        y_horizons = tf.stack(y_horizons)  # shape: (num_time_series, Batch_size, horizon)
        y_back_horizons = tf.stack(y_back_horizons)  # shape: (num_time_series, Batch_size, back_horizon)

        # Swap the axes to obtain the desired shape
        y_horizons = tf.transpose(y_horizons, perm=[1, 2, 0])  # shape: (Batch_size, horizon, num_time_series)
        y_back_horizons = tf.transpose(y_back_horizons,
                                       perm=[1, 2, 0])  # shape: (Batch_size, back_horizon, num_time_series)

        return y_horizons, y_back_horizons

    @property
    def intermediate_variables(self):
        intermediate_vars = []
        for block in self._blocks:
            intermediate_vars.append(block.intermediate_variables)
        return intermediate_vars


class MultiTrendBlock(MultiBlock):
    def __init__(self,
                 num_time_series,
                 horizon,
                 back_horizon,
                 p_degree=2,
                 n_neurons=256,
                 fc_layers=4,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(horizon, back_horizon,
                         [SimpleTrendBlock(horizon, back_horizon, p_degree, n_neurons, fc_layers, dropout_rate, name=f'_ts{i}') for i in
                          range(num_time_series)],
                         **kwargs)


class MultiSeasonalityBlock(MultiBlock):
    def __init__(self,
                 num_time_series,
                 horizon,
                 back_horizon,
                 n_neurons=256,
                 fc_layers=4,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(horizon, back_horizon,
                         [SimpleSeasonalityBlock(
                             horizon, back_horizon, n_neurons, fc_layers, dropout_rate, name=f'_ts{i}')
                          for i in range(num_time_series)],
                         **kwargs)
