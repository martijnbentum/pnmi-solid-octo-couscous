import numpy as np

from .metrics import evaluate_labels


def analyze_all_dummy_datasets():
    '''Run PNMI analysis across all dummy showcase datasets.

    returns:  Dictionary keyed by dataset name. Each value is the full result
              from `analyze_dummy_dataset(...)`, including the source labels.
    '''
    return {name: analyze_dummy_dataset(name) for name in dummy_pnmi_datasets()}


def analyze_dummy_dataset(name):
    '''Run PNMI analysis for one named dummy dataset.

    name:     One of `perfect`, `high`, `medium`, `low`, or `none`.

    returns:  Result dictionary from `evaluate_labels(...)` plus the dataset
              name and the source `phone_labels` and `cluster_labels`.
    '''
    datasets = dummy_pnmi_datasets()
    if name not in datasets:
        raise ValueError(f'unknown dummy dataset: {name}')

    phone_labels, cluster_labels = datasets[name]
    result = evaluate_labels(phone_labels, cluster_labels,
        return_diagnostics=True)
    result['name'] = name
    result['phone_labels'] = phone_labels
    result['cluster_labels'] = cluster_labels
    return result


def dummy_pnmi_datasets():
    '''Return named dummy datasets ordered by expected PNMI strength.

    returns:  Dictionary containing aligned `(phone_labels, cluster_labels)`
              pairs for `perfect`, `high`, `medium`, `low`, and `none`.
    '''
    return {
        'perfect': perfect_pnmi_data(),
        'high': high_pnmi_data(),
        'medium': medium_pnmi_data(),
        'low': low_pnmi_data(),
        'none': no_pnmi_data(),
    }


def perfect_pnmi_data():
    '''Return labels with a one-to-one phone-to-cluster mapping.

    returns:  Tuple `(phone_labels, cluster_labels)` as 1D NumPy arrays.
    '''
    phone_labels = (
        ['aa'] * 20 +
        ['bb'] * 20 +
        ['cc'] * 20 +
        ['dd'] * 20)
    cluster_labels = (
        ['c0'] * 20 +
        ['c1'] * 20 +
        ['c2'] * 20 +
        ['c3'] * 20)
    return _as_label_arrays(phone_labels, cluster_labels)


def high_pnmi_data():
    '''Return labels with strong but imperfect phone-cluster alignment.

    returns:  Tuple `(phone_labels, cluster_labels)` as 1D NumPy arrays.
    '''
    phone_labels = (
        ['aa'] * 20 +
        ['bb'] * 20 +
        ['cc'] * 20 +
        ['dd'] * 20)
    cluster_labels = (
        ['c0'] * 18 + ['c1'] * 2 +
        ['c0'] * 2 + ['c1'] * 18 +
        ['c2'] * 18 + ['c3'] * 2 +
        ['c2'] * 2 + ['c3'] * 18)
    return _as_label_arrays(phone_labels, cluster_labels)


def medium_pnmi_data():
    '''Return labels where clusters merge phone pairs deterministically.

    returns:  Tuple `(phone_labels, cluster_labels)` as 1D NumPy arrays.
    '''
    phone_labels = (
        ['aa'] * 20 +
        ['bb'] * 20 +
        ['cc'] * 20 +
        ['dd'] * 20)
    cluster_labels = (
        ['c0'] * 20 +
        ['c0'] * 20 +
        ['c1'] * 20 +
        ['c1'] * 20)
    return _as_label_arrays(phone_labels, cluster_labels)


def low_pnmi_data():
    '''Return labels with weak but non-zero phone-cluster correlation.

    returns:  Tuple `(phone_labels, cluster_labels)` as 1D NumPy arrays.
    '''
    phone_labels = (
        ['aa'] * 20 +
        ['bb'] * 20 +
        ['cc'] * 20 +
        ['dd'] * 20)
    cluster_labels = (
        ['c0'] * 8 + ['c1'] * 5 + ['c2'] * 4 + ['c3'] * 3 +
        ['c0'] * 3 + ['c1'] * 8 + ['c2'] * 5 + ['c3'] * 4 +
        ['c0'] * 4 + ['c1'] * 3 + ['c2'] * 8 + ['c3'] * 5 +
        ['c0'] * 5 + ['c1'] * 4 + ['c2'] * 3 + ['c3'] * 8)
    return _as_label_arrays(phone_labels, cluster_labels)


def no_pnmi_data():
    '''Return labels with identical cluster distribution for every phone.

    returns:  Tuple `(phone_labels, cluster_labels)` as 1D NumPy arrays.
    '''
    phone_labels = (
        ['aa'] * 20 +
        ['bb'] * 20 +
        ['cc'] * 20 +
        ['dd'] * 20)
    cluster_labels = ['c0', 'c1', 'c2', 'c3'] * 20
    return _as_label_arrays(phone_labels, cluster_labels)


def _as_label_arrays(phone_labels, cluster_labels):
    phone_labels = np.asarray(phone_labels, dtype=object)
    cluster_labels = np.asarray(cluster_labels, dtype=object)
    if phone_labels.ndim != 1:
        raise ValueError('phone_labels must be a 1D sequence')
    if cluster_labels.ndim != 1:
        raise ValueError('cluster_labels must be a 1D sequence')
    if phone_labels.size == 0 or cluster_labels.size == 0:
        raise ValueError('phone_labels and cluster_labels must not be empty')
    if phone_labels.size != cluster_labels.size:
        raise ValueError(
            'phone_labels and cluster_labels must have the same length')
    return phone_labels, cluster_labels
