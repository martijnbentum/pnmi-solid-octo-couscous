import numpy as np

from .metrics import evaluate_labels


def labels_from_count_matrix(count_matrix):
    '''Build aligned frame labels from a phone-by-cluster count matrix.

    count_matrix: rows are phone labels, columns are cluster labels
    '''
    count_matrix = np.asarray(count_matrix, dtype=int)
    if count_matrix.ndim != 2:
        raise ValueError('count_matrix must be a 2D array')
    if np.any(count_matrix < 0):
        raise ValueError('count_matrix must be non-negative')
    if count_matrix.sum() == 0:
        raise ValueError('count_matrix must contain at least one frame')

    phone_labels = []
    cluster_labels = []
    for phone_index in range(count_matrix.shape[0]):
        for cluster_index in range(count_matrix.shape[1]):
            count = int(count_matrix[phone_index, cluster_index])
            if count == 0: continue
            phone_labels.extend([phone_index] * count)
            cluster_labels.extend([cluster_index] * count)

    return np.asarray(phone_labels, dtype=int), np.asarray(cluster_labels,
        dtype=int)


def perfect_pnmi_data():
    '''Return labels with a one-to-one phone-to-cluster mapping.'''
    count_matrix = np.array(
        [
            [20, 0, 0, 0],
            [0, 20, 0, 0],
            [0, 0, 20, 0],
            [0, 0, 0, 20],
        ],
        dtype=int)
    return labels_from_count_matrix(count_matrix)


def high_pnmi_data():
    '''Return labels with strong but imperfect phone-cluster alignment.'''
    count_matrix = np.array(
        [
            [18, 2, 0, 0],
            [2, 18, 0, 0],
            [0, 0, 18, 2],
            [0, 0, 2, 18],
        ],
        dtype=int)
    return labels_from_count_matrix(count_matrix)


def medium_pnmi_data():
    '''Return labels where clusters merge phone pairs deterministically.'''
    count_matrix = np.array(
        [
            [20, 0],
            [20, 0],
            [0, 20],
            [0, 20],
        ],
        dtype=int)
    return labels_from_count_matrix(count_matrix)


def low_pnmi_data():
    '''Return labels with weak but non-zero phone-cluster correlation.'''
    count_matrix = np.array(
        [
            [8, 5, 4, 3],
            [3, 8, 5, 4],
            [4, 3, 8, 5],
            [5, 4, 3, 8],
        ],
        dtype=int)
    return labels_from_count_matrix(count_matrix)


def no_pnmi_data():
    '''Return labels with identical cluster distribution for every phone.'''
    count_matrix = np.array(
        [
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        dtype=int)
    return labels_from_count_matrix(count_matrix)


def dummy_pnmi_datasets():
    '''Return named dummy datasets ordered by expected PNMI strength.'''
    return {
        'perfect': perfect_pnmi_data(),
        'high': high_pnmi_data(),
        'medium': medium_pnmi_data(),
        'low': low_pnmi_data(),
        'none': no_pnmi_data(),
    }


def analyze_dummy_dataset(name):
    '''Run PNMI analysis for one named dummy dataset.'''
    datasets = dummy_pnmi_datasets()
    if name not in datasets:
        raise ValueError(f'unknown dummy dataset: {name}')

    phone_labels, cluster_labels = datasets[name]
    result = evaluate_labels(phone_labels, cluster_labels,
        return_diagnostics=True)
    result['name'] = name
    return result


def analyze_all_dummy_datasets():
    '''Run PNMI analysis across all named dummy datasets.'''
    return {
        name: analyze_dummy_dataset(name)
        for name in dummy_pnmi_datasets()
    }
