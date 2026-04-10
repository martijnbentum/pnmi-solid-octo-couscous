import numpy as np


def _as_1d_labels(labels):
    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    return array


def _is_invalid(value, invalid_label):
    if invalid_label is None:
        return False
    if isinstance(invalid_label, float) and np.isnan(invalid_label):
        return isinstance(value, float) and np.isnan(value)
    return value == invalid_label


def filter_valid_frames(phone_labels, cluster_labels, invalid_label=None):
    '''Filter aligned labels and drop invalid frames.

    invalid_label: value to ignore in either label stream. If None, all
    frames are retained.
    '''
    phone_labels = _as_1d_labels(phone_labels)
    cluster_labels = _as_1d_labels(cluster_labels)

    if phone_labels.size == 0:
        raise ValueError('phone_labels and cluster_labels must not be empty')
    if phone_labels.size != cluster_labels.size:
        raise ValueError(
            'phone_labels and cluster_labels must have the same length'
        )

    if invalid_label is None:
        return phone_labels, cluster_labels

    valid_indices = []
    paired_labels = zip(phone_labels.tolist(), cluster_labels.tolist())
    for index, (phone_label, cluster_label) in enumerate(paired_labels):
        if _is_invalid(phone_label, invalid_label): continue
        if _is_invalid(cluster_label, invalid_label): continue
        valid_indices.append(index)

    if not valid_indices:
        raise ValueError('no valid frames remain after filtering')

    return phone_labels[valid_indices], cluster_labels[valid_indices]


def _inverse_indices(labels):
    label_to_index = {}
    inverse = np.empty(labels.size, dtype=int)

    for index, label in enumerate(labels.tolist()):
        mapped = label_to_index.get(label)
        if mapped is None:
            mapped = len(label_to_index)
            label_to_index[label] = mapped
        inverse[index] = mapped

    return inverse, len(label_to_index)


def _count_matrix(phone_labels, cluster_labels):
    phone_labels, cluster_labels = filter_valid_frames(
        phone_labels,
        cluster_labels,
    )
    phone_inverse, n_phones = _inverse_indices(phone_labels)
    cluster_inverse, n_clusters = _inverse_indices(cluster_labels)

    counts = np.zeros((n_phones, n_clusters), dtype=float)
    np.add.at(counts, (phone_inverse, cluster_inverse), 1.0)
    return counts


def joint_distribution(phone_labels, cluster_labels):
    '''Estimate the joint distribution p(phone, cluster).'''
    counts = _count_matrix(phone_labels, cluster_labels)
    return counts / counts.sum()


def marginals(phone_labels, cluster_labels):
    '''Compute p(phone) and p(cluster).'''
    joint = joint_distribution(phone_labels, cluster_labels)
    return joint.sum(axis=1), joint.sum(axis=0)


def entropy(probabilities):
    '''Compute Shannon entropy from a probability vector or matrix.'''
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def mutual_information(phone_labels, cluster_labels):
    '''Compute mutual information between phone and cluster labels.'''
    joint = joint_distribution(phone_labels, cluster_labels)
    phone_marginal = joint.sum(axis=1, keepdims=True)
    cluster_marginal = joint.sum(axis=0, keepdims=True)
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
    return float(counts.max(axis=0).sum() / counts.sum())


def cluster_purity(phone_labels, cluster_labels):
    '''Compute average cluster purity within phone labels.'''
    counts = _count_matrix(phone_labels, cluster_labels)
    return float(counts.max(axis=1).sum() / counts.sum())


def evaluate_labels(phone_labels, cluster_labels, invalid_label=None,
    return_diagnostics=False):
    '''Compute PNMI metrics for one aligned discrete label stream.'''
    phone_labels, cluster_labels = filter_valid_frames(
        phone_labels,
        cluster_labels,
        invalid_label=invalid_label)
    joint = joint_distribution(phone_labels, cluster_labels)
    phone_marginal = joint.sum(axis=1)
    cluster_marginal = joint.sum(axis=0)
    phone_entropy = entropy(phone_marginal)
    cluster_entropy = entropy(cluster_marginal)
    information = mutual_information(phone_labels, cluster_labels)

    result = {
        'valid_frame_count': int(phone_labels.size),
        'n_phone_labels': int(phone_marginal.size),
        'n_cluster_labels': int(cluster_marginal.size),
        'mutual_information': information,
        'phone_entropy': phone_entropy,
        'cluster_entropy': cluster_entropy,
        'pnmi': 0.0 if phone_entropy == 0.0 else information / phone_entropy,
        'phone_purity': phone_purity(phone_labels, cluster_labels),
        'cluster_purity': cluster_purity(phone_labels, cluster_labels),
    }

    if return_diagnostics:
        result['mi_over_phone_entropy'] = (
            0.0 if phone_entropy == 0.0 else information / phone_entropy
        )
        result['mi_over_cluster_entropy'] = (
            0.0 if cluster_entropy == 0.0 else information / cluster_entropy
        )
        result['joint_distribution'] = joint
        result['phone_marginal'] = phone_marginal
        result['cluster_marginal'] = cluster_marginal

    return result
