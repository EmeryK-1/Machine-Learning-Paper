import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_nbeats(y_pred, y_residuals, y_true=None):
    """
    For every time series (y_pred.shape[1]), make subplots (1+len(y_residuals)). The first subplot plots y_pred and y_true if given. The other subplots plot the residuals.
    :param y_pred: Tensor of shape (n, p)
    :param y_residuals: list of tensors of shape (n, p)
    :param y_true:
    :return:
    """
    y_pred = y_pred[0, :,:]
    y_residuals = [a[0,:,:] for a in y_residuals]
    p = y_pred.shape[1]
    # Create figure with p columns and 1+len(y_residuals) rows
    fig, axs = plt.subplots(p, 1+len(y_residuals), figsize=(20, 10))
    for i in range(p):
        # Plot the prediction
        axs[i, 0].plot(y_pred[:, i])
        # Title subplot
        axs[i, 0].set_title(f'ts {i}')

        if y_true is not None:
            axs[i, 0].plot(y_true[:, i])
            # Add legend
            axs[i, 0].legend(['pred', 'true'])
        # Plot the residuals
        for j in range(len(y_residuals)):
            axs[i, j+1].plot(y_residuals[j][:, i])
            # Title subplot with the name of the residual
            axs[i, j+1].set_title(f'ts {i} - stack {j}')
        # Plot the sum of the residuals over the prediction

    plt.show()
    return fig