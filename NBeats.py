import tensorflow as tf


class N_BEATS(tf.keras.Model):
    def __init__(self,
                 stacks,
                 **kwargs):
        super().__init__(**kwargs)
        self._stacks = stacks
        self._residuals_y = tf.TensorArray(tf.float32, size=(len(
            self._stacks)))  # Stock trend and seasonality curves during inference


    def call(self, inputs):
        y_horizon = 0.
        for idx, stack in enumerate(self._stacks):
            residual_y, inputs = stack(inputs)
            self._residuals_y = self._residuals_y.write(idx, residual_y)
            y_horizon = tf.add(y_horizon, residual_y)
        return y_horizon

    def explain_forecast(self, inputs):
        y_horizon = 0.
        residuals_y = []
        for idx, stack in enumerate(self._stacks):
            residual_y, inputs = stack(inputs)
            residuals_y.append(residual_y)
            y_horizon = tf.add(y_horizon, residual_y)
        return y_horizon, residuals_y

    @property
    def intermediate_variables(self):
        intermediate_vars = []
        for stack in self._stacks:
            intermediate_vars.append(stack.intermediate_variables)
        return intermediate_vars


class Stack(tf.keras.layers.Layer):
    def __init__(self, blocks, **kwargs):
        super().__init__(**kwargs)

        self._blocks = blocks

    def call(self, inputs):
        y_horizon = 0.
        for block in self._blocks:
            residual_y, y_back_horizon = block(
                inputs)  # shape: (Batch_size, horizon), (Batch_size, back_horizon)
            inputs = tf.subtract(inputs, y_back_horizon)
            y_horizon = tf.add(y_horizon, residual_y)  # shape: (Batch_size, horizon)
        return y_horizon, inputs

    @property
    def intermediate_variables(self):
        intermediate_vars = []
        for block in self._blocks:
            intermediate_vars.append(block.intermediate_variables)
        return intermediate_vars
