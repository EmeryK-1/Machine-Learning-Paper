from tensorflow import cast, reshape, range, concat, sin, cos, reduce_sum
from tensorflow.python.keras.layers import Layer, Dense, Dropout
import numpy as np


class SimpleTrendBlock(Layer):
    def __init__(self,
                 horizon,
                 back_horizon,
                 p_degree=2,
                 n_neurons=256,
                 fc_layers=4,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._p_degree = reshape(range(p_degree + 1, dtype='float32'),
                                 shape=(-1, 1))  # Shape (-1, 1) in order to broadcast horizon to all p degrees
        self._horizon = cast(horizon, dtype='float32')
        self._back_horizon = cast(back_horizon, dtype='float32')
        self._n_neurons = n_neurons

        self._FC_stack = [Dense(n_neurons, activation='relu', kernel_initializer="glorot_uniform") for _ in range(fc_layers)]

        self._dropout = Dropout(dropout_rate)

        self._FC_back_horizon = self.add_weight(shape=(n_neurons, p_degree + 1),
                                                trainable=True,
                                                initializer="glorot_uniform",
                                                name='FC_back_horizon_trend'+self.name)

        self._FC_horizon = self.add_weight(shape=(n_neurons, p_degree + 1),
                                           trainable=True,
                                           initializer="glorot_uniform",
                                           name='FC_horizon_trend'+self.name)

        self._horizon_coef = (range(self._horizon) / self._horizon) ** self._p_degree  # shape: (horizon, p_degree+1)
        self._back_horizon_coef = (range(
            self._back_horizon) / self._back_horizon) ** self._p_degree  # shape: (back_horizon, p_degree+1)

        self.theta_horizon = None
        self.theta_back_horizon = None

    def call(self, inputs, training=False):
        for dense in self._FC_stack:
            inputs = dense(inputs)  # shape: (Batch_size, n_neurons)
            inputs = self._dropout(inputs, training=True)  # We bind first layers by a dropout

        theta_back_horizon = inputs @ self._FC_back_horizon  # shape: (Batch_size, p_degree+1)
        theta_horizon = inputs @ self._FC_horizon  # shape: (Batch_size, p_degree+1)
        if not training:
            self.theta_back_horizon = theta_back_horizon
            self.theta_horizon = theta_horizon
        y_back_horizon = theta_back_horizon @ self._back_horizon_coef  # shape: (Batch_size, back_horizon)
        y_horizon = theta_horizon @ self._horizon_coef  # shape: (Batch_size, horizon)

        return y_horizon, y_back_horizon

    @property
    def intermediate_variables(self):
        return {'theta_horizon': self.theta_horizon,
                'theta_back_horizon': self.theta_back_horizon}


class SimpleSeasonalityBlock(Layer):
    def __init__(self,
                 horizon,
                 back_horizon,
                 n_neurons=256,
                 fc_layers=4,
                 dropout_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        horizon_fourier_order = horizon
        back_horizon_fourier_order = back_horizon

        self._horizon = horizon
        self._back_horizon = back_horizon
        self._horizon_fourier_order = reshape(range(horizon_fourier_order, dtype='float32'),
                                              shape=(-1, 1))  # Broadcast horizon on multiple fourier order
        self._back_horizon_fourier_order = reshape(range(back_horizon_fourier_order, dtype='float32'),
                                                   shape=(-1, 1))  # Broadcast horizon on multiple fourier order

        # Workout the number of neurons needed to compute seasonality coefficients
        horizon_neurons = reduce_sum(2 * horizon_fourier_order)
        back_horizon_neurons = reduce_sum(2 * back_horizon_fourier_order)

        self._FC_stack = [Dense(n_neurons, activation='relu', kernel_initializer="glorot_uniform") for _ in range(fc_layers)]

        self._dropout = Dropout(dropout_rate)

        self._FC_back_horizon = self.add_weight(shape=(n_neurons, back_horizon_neurons),
                                                trainable=True,
                                                initializer="glorot_uniform",
                                                name='FC_back_horizon_seasonality'+self.name)

        self._FC_horizon = self.add_weight(shape=(n_neurons, horizon_neurons),
                                           trainable=True,
                                           initializer="glorot_uniform",
                                           name='FC_horizon_seasonality'+self.name)

        # Workout cos and sin seasonality coefficents
        time_horizon = range(self._horizon, dtype='float32') / self._horizon
        horizon_seasonality = 2 * np.pi * self._horizon_fourier_order * time_horizon
        self._horizon_coef = concat((cos(horizon_seasonality),
                                     sin(horizon_seasonality)), axis=0)

        time_back_horizon = range(self._back_horizon, dtype='float32') / self._back_horizon
        back_horizon_seasonality = 2 * np.pi * self._back_horizon_fourier_order * time_back_horizon
        self._back_horizon_coef = concat((cos(back_horizon_seasonality),
                                          sin(back_horizon_seasonality)), axis=0)

        self.theta_horizon = None
        self.theta_back_horizon = None

    def call(self, inputs, training=False):
        for dense in self._FC_stack:
            inputs = dense(inputs)  # shape: (Batch_size, nb_neurons)
            inputs = self._dropout(inputs, training=True)  # We bind first layers by a dropout

        theta_horizon = inputs @ self._FC_horizon  # shape: (Batch_size, 2 * fourier order)
        theta_back_horizon = inputs @ self._FC_back_horizon  # shape: (Batch_size, 2 * fourier order)

        if not training:
            self.theta_back_horizon = theta_back_horizon
            self.theta_horizon = theta_horizon

        y_horizon = theta_horizon @ self._horizon_coef  # shape: (Batch_size, horizon)
        y_back_horizon = theta_back_horizon @ self._back_horizon_coef  # shape: (Batch_size, back_horizon)

        return y_horizon, y_back_horizon


    @property
    def intermediate_variables(self):
        return {'theta_horizon': self.theta_horizon,
                'theta_back_horizon': self.theta_back_horizon}
