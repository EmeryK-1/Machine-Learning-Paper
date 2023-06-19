import numpy as np
import pandas as pd

def to_companion(coefficients):
    m = coefficients.shape[0]
    p = coefficients.shape[1] // m
    # Stack coeffs and i, fill rest with 0
    coefficients = np.vstack((coefficients, np.zeros((m*(p-1), m*p))))
    # Set bottom left (p-1)x(p-1) square to be identity
    coefficients[m:, :m*(p-1)] = np.eye(m*(p-1))
    return coefficients


def make_stable(companion_matrix, num_time_series, num_lags):
    while np.max(np.abs(np.linalg.eigvals(companion_matrix))) > 1:
        companion_matrix = companion_matrix * 0.95
        companion_matrix[num_time_series:, :num_time_series*(num_lags-1)] = np.eye(num_time_series*(num_lags-1))
    return companion_matrix


def simulate(initial_state, companion_matrix, n, noise=False):
    """
    :param initial_state: pxm array containing the first p lags needed for computation for each time series
    :param companion_matrix: p*m x p*m VAR companion matrix
    :param n: amount of time steps to generate
    :return: simulated data
    """
    if noise is True:
        noise = np.random.normal(size=(n, initial_state.shape[1]))
    m = initial_state.shape[1]
    p = companion_matrix.shape[0]//m
    initial_state = initial_state.flatten()
    sim = [initial_state[m*i:m*(i+1)] for i in range(p)]
    for i in range(n):
        initial_state = np.dot(companion_matrix, initial_state)
        initial_state[:m] = initial_state[:m] + (np.random.normal(size=m) if noise is False else noise[i])
        sim.append(initial_state[:m])
    if noise is False:
        return np.array(sim)
    return np.array(sim), noise


def generate_trend(time_steps, n_variables, trend_slopes):
    """
    Generate trend for multivariate time series.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.
    trend_slopes (float or list): The slopes of the trends for each variable.

    Returns:
    pd.DataFrame: A DataFrame with the generated trend.
    """
    if trend_slopes is None:
        return pd.DataFrame(np.zeros((time_steps, n_variables)), columns=['Variable_{}'.format(i) for i in range(n_variables)])
    # Check if trend_slopes is a number or a list
    if isinstance(trend_slopes, (int, float)):
        trend_slopes = [trend_slopes] * n_variables

    # Initialize an empty DataFrame
    df = pd.DataFrame(index=range(time_steps), columns=['Variable_{}'.format(i) for i in range(n_variables)])

    # Generate a trend for each variable
    for i in range(n_variables):
        df['Variable_{}'.format(i)] = np.linspace(start=0, stop=trend_slopes[i] * time_steps, num=time_steps)

    return df


def generate_seasonality(time_steps, n_variables, seasonality_scales, seasonality_frequencies):
    """
    Generate seasonality for multivariate time series.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.
    seasonality_scales (float or list): The scales of the seasonality for each variable.
    seasonality_frequencies (float or list): The frequencies of the seasonality for each variable.

    Returns:
    pd.DataFrame: A DataFrame with the generated seasonality.
    """
    if seasonality_scales is None or seasonality_frequencies is None:
        return pd.DataFrame(np.zeros((time_steps, n_variables)), columns=['Variable_{}'.format(i) for i in range(n_variables)])

    # Check if seasonality_scales and seasonality_frequencies are numbers or lists
    if isinstance(seasonality_scales, (int, float)):
        seasonality_scales = [seasonality_scales] * n_variables
    if isinstance(seasonality_frequencies, (int, float)):
        seasonality_frequencies = [seasonality_frequencies] * n_variables

    # Initialize an empty DataFrame
    df = pd.DataFrame(index=range(time_steps), columns=['Variable_{}'.format(i) for i in range(n_variables)])

    # Generate seasonality for each variable
    for i in range(n_variables):
        df['Variable_{}'.format(i)] = seasonality_scales[i] * np.sin(np.linspace(start=0, stop=2*np.pi*seasonality_frequencies[i], num=time_steps))

    return df


def generate_interrelations(time_steps, n_variables, companion_matrix, noise=False):
    """
    Simulate a Vector Autoregressive (VAR) model.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.
    companion_matrix (np.ndarray): The VAR companion matrix.
    noise (bool or np.ndarray): Whether to add random noise, or a specific noise array.

    Returns:
    pd.DataFrame: A DataFrame with the generated interrelations.
    """
    # If companion matrix is none, return 0s
    if companion_matrix is None:
        interrelations = pd.DataFrame(np.zeros((time_steps, n_variables)), columns=['Variable_{}'.format(i) for i in range(n_variables)])
        return interrelations, np.zeros((time_steps, n_variables)) if isinstance(noise, bool) and noise else None
    # Determine the number of lags from the companion matrix
    p = companion_matrix.shape[0] // n_variables

    # Create the initial state with zeros
    initial_state = np.zeros(p * n_variables)

    # Prepare for simulation
    interrelations = [initial_state[n_variables * i: n_variables * (i + 1)] for i in range(p)]
    if isinstance(noise, bool) and noise:
        noise = np.random.normal(size=(time_steps, n_variables))

    # Simulate the time series
    for i in range(time_steps-p):
        initial_state = np.dot(companion_matrix, initial_state)
        initial_state[:n_variables] += np.random.normal(size=n_variables) if isinstance(noise, bool) else noise[i]
        interrelations.append(initial_state[:n_variables])

    interrelations = pd.DataFrame(np.array(interrelations), columns=['Variable_{}'.format(i) for i in range(n_variables)])

    return interrelations, None if isinstance(noise, bool) and not noise else noise


def generate_random_walk(time_steps, n_variables):
    """
    Generate random walk for multivariate time series.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.

    Returns:
    pd.DataFrame: A DataFrame with the generated random walk.
    """
    random_walk = np.cumsum(np.random.randn(time_steps, n_variables), axis=0)
    random_walk = pd.DataFrame(random_walk, columns=['Variable_{}'.format(i) for i in range(n_variables)])
    return random_walk


def generate_multivariate_time_series(time_steps, n_variables, trend_slope, seasonality_scale, seasonality_frequency, companion_matrix, random_walk=True, noise=False):
    """
    Generate multivariate time series data with trend, seasonality, interrelations, and random walk.

    Parameters:
    time_steps (int): The number of time steps.
    n_variables (int): The number of variables.
    trend_slope (float): The slope of the trend.
    seasonality_scale (float): The scale of the seasonality.
    seasonality_frequency (int): The frequency of the seasonality.
    companion_matrix (np.ndarray): The VAR companion matrix.
    noise (bool or np.ndarray): Whether to add random noise, or a specific noise array.

    Returns:
    pd.DataFrame: A DataFrame with the generated time series.
    """
    # Generate each component
    trend = generate_trend(time_steps, n_variables, trend_slope)
    seasonality = generate_seasonality(time_steps, n_variables, seasonality_scale, seasonality_frequency)
    interrelations, _ = generate_interrelations(time_steps, n_variables, companion_matrix, noise)

    # Combine all components
    time_series = trend + seasonality + interrelations
    if random_walk:
        time_series += generate_random_walk(time_steps, n_variables)

    return time_series


def generate_simulations(n_simulations,n_variables, time_steps, var_lag=2):
    simulations = []

    for _ in range(n_simulations):
        # Randomly generate parameters
        trend_slopes = np.random.normal(0, 1, n_variables)
        seasonality_scales = np.random.uniform(0.5, 5, n_variables)
        seasonality_frequencies = np.random.randint(max(1, time_steps/100), time_steps/5, size=n_variables)
        companion_matrix = np.random.normal(size=(n_variables, n_variables * var_lag))
        companion_matrix = make_stable(to_companion(companion_matrix), n_variables, var_lag)

        # Generate individual components
        trend = generate_trend(time_steps, n_variables, trend_slopes)
        seasonality = generate_seasonality(time_steps, n_variables, seasonality_scales, seasonality_frequencies)
        interrelations, _ = generate_interrelations(time_steps, n_variables, companion_matrix)
        random_walk = generate_random_walk(time_steps, n_variables)

        # Combine the components to form the 10 combinations
        data = {
            'ir': interrelations,
            't_ir': trend + interrelations,
            's_ir': seasonality + interrelations,
            't_s_ir': trend + seasonality + interrelations,
            't_s': trend + seasonality,
            'rw_ir': random_walk + interrelations,
            't_rw_ir': trend + random_walk + interrelations,
            's_rw_ir': seasonality + random_walk + interrelations,
            't_s_rw_ir': trend + seasonality + random_walk + interrelations,
            'rw': random_walk
        }

        # Store arguments and data in a dictionary for each simulation
        simulations.append({
            'arguments': {
                'trend_slopes': trend_slopes,
                'seasonality_scales': seasonality_scales,
                'seasonality_frequencies': seasonality_frequencies,
                'companion_matrix': companion_matrix,
            },
            'data': data,
        })

    return simulations


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Define parameters
    time_steps = 1000
    n_variables = 3
    trend_slope = [0.01, 0.1, -0.1]
    seasonality_scale = 4.0
    seasonality_frequency = [4, 8, 16]

    # Generate random states
    companion_matrix = np.random.normal(size=(n_variables, n_variables * 2))
    # Compute companion
    companion_matrix = make_stable(to_companion(companion_matrix), n_variables, 2)
    print(companion_matrix)

    # Generate the time series
    time_series = generate_multivariate_time_series(
        time_steps=time_steps,
        n_variables=n_variables,
        trend_slope=trend_slope,
        seasonality_scale=seasonality_scale,
        seasonality_frequency=seasonality_frequency,
        companion_matrix=companion_matrix,
        random_walk=True
    )

    # Display the result
    time_series.plot()
    plt.show()