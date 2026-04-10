import numpy as np


def _as_1d_labels(labels):
    array = np.asarray(labels)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    return array


def _validate_labels(phone_labels, cluster_labels):
    phone_labels = _as_1d_labels(phone_labels)
    cluster_labels = _as_1d_labels(cluster_labels)

    if phone_labels.size == 0:
        raise ValueError('phone_labels and cluster_labels must not be empty')
    if phone_labels.size != cluster_labels.size:
        raise ValueError(
            'phone_labels and cluster_labels must have the same length'
        )
    return phone_labels, cluster_labels


def _inverse_indices(labels):
    label_to_index = {}
    inverse = np.empty(labels.size, dtype = int)

    for index, label in enumerate(labels.tolist()):
        mapped = label_to_index.get(label)
        if mapped is None:
            mapped = len(label_to_index)
            label_to_index[label] = mapped
        inverse[index] = mapped

    return inverse, len(label_to_index)


def _count_matrix(phone_labels, cluster_labels):
    phone_labels, cluster_labels = _validate_labels(
        phone_labels,
        cluster_labels,
    )
    phone_inverse, n_phones = _inverse_indices(phone_labels)
    cluster_inverse, n_clusters = _inverse_indices(cluster_labels)

    counts = np.zeros((n_phones, n_clusters), dtype = float)
    np.add.at(counts, (phone_inverse, cluster_inverse), 1.0)
    return counts


def joint_distribution(phone_labels, cluster_labels):
    '''Estimate the joint distribution p(phone, cluster).'''
    counts = _count_matrix(phone_labels, cluster_labels)
    return counts / counts.sum()


def marginals(phone_labels, cluster_labels):
    '''Compute p(phone) and p(cluster).'''
    joint = joint_distribution(phone_labels, cluster_labels)
    return joint.sum(axis = 1), joint.sum(axis = 0)


def entropy(probabilities):
    '''Compute Shannon entropy from a probability vector or matrix.'''
    probabilities = np.asarray(probabilities, dtype = float)
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def mutual_information(phone_labels, cluster_labels):
    '''Compute mutual information between phone and cluster labels.'''
    joint = joint_distribution(phone_labels, cluster_labels)
    phone_marginal = joint.sum(axis = 1, keepdims = True)
    cluster_marginal = joint.sum(axis = 0, keepdims = True)
    denominator = phone_marginal * cluster_marginal
    mask = joint > 0.0
    values = joint[mask] * np.log(joint[mask] / denominator[mask])
    return float(values.sum())


def pnmi(phone_labels, cluster_labels):
    '''Compute phone-normalized mutual information.'''
    phone_marginal, _ = marginals(phone_labels, cluster_labels)
    phone_entropy = entropy(phone_marginal)
    if phone_entropy == 0.0:
        return 0.0
    return mutual_information(phone_labels, cluster_labels) / phone_entropy


def phone_purity(phone_labels, cluster_labels):
    '''Compute average phone purity within cluster labels.'''
    counts = _count_matrix(phone_labels, cluster_labels)
    return float(counts.max(axis = 0).sum() / counts.sum())


def cluster_purity(phone_labels, cluster_labels):
    '''Compute average cluster purity within phone labels.'''
    counts = _count_matrix(phone_labels, cluster_labels)
    return float(counts.max(axis = 1).sum() / counts.sum())
