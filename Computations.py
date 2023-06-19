import numpy as np


def resize_companion_matrices(inferred, real):
    if inferred.shape[1] == real.shape[1] and inferred.shape[0] == real.shape[0]:
        return inferred, real
    new_inferred, num_variables, lags = make_companion(inferred)
    if real.shape[1] > num_variables * lags:
        new_inferred = np.zeros(real.shape)
        new_inferred[:inferred.shape[0], :inferred.shape[1]] = inferred
        new_inferred, num_variables, lags = make_companion(new_inferred)
    real = expand_companion(real, num_variables, lags)
    return new_inferred, real


def make_companion(matrix):
    companion = np.zeros((matrix.shape[1], matrix.shape[1]))
    num_variables = int(matrix.shape[0])
    lags = int(matrix.shape[1] / num_variables)
    companion[num_variables:, :num_variables * (lags - 1)] = np.eye(num_variables * (lags - 1))
    companion[:num_variables, :] = matrix
    return companion, num_variables, lags


def expand_companion(matrix, num_variables, lags):
    companion = np.zeros((num_variables * lags, num_variables * lags))
    companion[num_variables:, :num_variables * (lags - 1)] = np.eye(num_variables * (lags - 1))
    companion[:num_variables, :matrix.shape[1]] = matrix[:num_variables, :]
    return companion


def compare_frobenius(companion_matrix1, companion_matrix2):
    """
    Compare Frobenius norm of two companion matrices
    """
    difference = np.linalg.norm(companion_matrix1 - companion_matrix2, ord='fro')
    return difference


def compare_cosine(companion_matrix1, companion_matrix2):
    """
    Compare cosine similarity of two companion matrices
    """
    companion_matrix1 = companion_matrix1.flatten()
    companion_matrix2 = companion_matrix2.flatten()

    similarity = np.dot(companion_matrix1, companion_matrix2) / (
            np.linalg.norm(companion_matrix1) * np.linalg.norm(companion_matrix2))

    return similarity


def compare_spectral_radius(companion_matrix1, companion_matrix2):
    """
    Compare the spectral radius of two companion matrices
    """
    spectral_radius1 = np.max(np.abs(np.linalg.eigvals(companion_matrix1)))
    spectral_radius2 = np.max(np.abs(np.linalg.eigvals(companion_matrix2)))

    # Calculate the difference in spectral radii
    difference = np.abs(spectral_radius1 - spectral_radius2)

    return difference


def compare_eigenvalues(companion_matrix1, companion_matrix2):
    # Fill nan values with 0
    """
    Compare the eigenvalues of two companion matrices
    """
    if companion_matrix1[:3].sum().sum()==0 and companion_matrix2[:3].sum().sum()==0:
        return 0
    if companion_matrix1[:3].sum().sum()==0:
        return np.sum(np.abs(np.linalg.eigvals(companion_matrix2)))
    if companion_matrix2[:3].sum().sum()==0:
        return np.sum(np.abs(np.linalg.eigvals(companion_matrix1)))
    eigenvalues1 = np.linalg.eigvals(companion_matrix1)
    eigenvalues2 = np.linalg.eigvals(companion_matrix2)
    eigenvalues2 = np.sort(eigenvalues2)
    eigenvalues1 = np.sort(eigenvalues1)
    # Calculate the difference in eigenvalues
    difference = np.abs(eigenvalues1 - eigenvalues2)

    return difference.sum()