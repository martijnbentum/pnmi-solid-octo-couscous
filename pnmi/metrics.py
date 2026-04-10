import numpy as np


def evaluate_labels(phone_labels, cluster_labels, invalid_label=None,
    return_diagnostics=False):
    '''Compute PNMI and related statistics for one aligned label stream.

    This is the main entry point for HuBERT-style evaluation where each
    frame has one phone label and one discrete unit label. The function
    filters invalid frames when requested, estimates the empirical joint
    distribution, and derives mutual information, entropies, PNMI, phone
    purity, and cluster purity.

    phone_labels:        1D sequence of frame-level phone labels. Labels may
                         be strings, integers, or any hashable discrete values.
    cluster_labels:      1D sequence of frame-level cluster or unit labels with
                         the same number of frames as `phone_labels`.
    invalid_label:       Optional sentinel value to ignore in either stream.
                         Frames where the phone label or cluster label matches
                         this value are removed before computing the metrics.
    return_diagnostics:  If True, also include the joint distribution,
                         marginals, and entropy-normalized MI diagnostics.

    returns:             Dictionary with the fields `valid_frame_count`,
                         `n_phone_labels`, `n_cluster_labels`,
                         `mutual_information`, `phone_entropy`,
                         `cluster_entropy`, `pnmi`, `phone_purity`, and
                         `cluster_purity`. When `return_diagnostics=True`,
                         the result also includes `joint_distribution`,
                         `phone_marginal`, `cluster_marginal`,
                         `mi_over_phone_entropy`, and
                         `mi_over_cluster_entropy`.
    '''
    phone_labels, cluster_labels = filter_valid_frames(phone_labels,
        cluster_labels, invalid_label=invalid_label)
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
            0.0 if phone_entropy == 0.0 else information / phone_entropy)
        result['mi_over_cluster_entropy'] = (
            0.0 if cluster_entropy == 0.0 else information / cluster_entropy)
        result['joint_distribution'] = joint
        result['phone_marginal'] = phone_marginal
        result['cluster_marginal'] = cluster_marginal

    return result


def pnmi(phone_labels, cluster_labels):
    '''Compute phone-normalized mutual information for one label stream.

    This is the scalar metric most often reported for frame-level unit
    quality. It computes the mutual information between phone labels and
    discrete unit labels, then normalizes by the phone-label entropy.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        Float PNMI score. A value near 1 indicates that the unit
                    labels preserve most phone-level information. A value near
                    0 indicates little or no phone information.
    '''
    phone_marginal, _ = marginals(phone_labels, cluster_labels)
    phone_entropy = entropy(phone_marginal)
    if phone_entropy == 0.0: return 0.0
    return mutual_information(phone_labels, cluster_labels) / phone_entropy


def joint_distribution(phone_labels, cluster_labels):
    '''Estimate the empirical joint distribution p(phone, cluster).

    The distribution is derived from aligned frame-level labels after input
    validation. Rows correspond to distinct phone labels and columns
    correspond to distinct cluster labels in order of first appearance.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        2D NumPy array whose entries sum to 1.0.
    '''
    counts = _count_matrix(phone_labels, cluster_labels)
    return counts / counts.sum()


def phone_purity(phone_labels, cluster_labels):
    '''Compute average phone purity within cluster labels.

    For each cluster, this keeps the most frequent phone assignment and sums
    those maxima across clusters. The result is normalized by the total number
    of valid frames.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        Float purity score in the range [0, 1].
    '''
    counts = _count_matrix(phone_labels, cluster_labels)
    return float(counts.max(axis=0).sum() / counts.sum())


def cluster_purity(phone_labels, cluster_labels):
    '''Compute average cluster purity within phone labels.

    For each phone label, this keeps the most frequent cluster assignment and
    sums those maxima across phone classes. The result is normalized by the
    total number of valid frames.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        Float purity score in the range [0, 1].
    '''
    counts = _count_matrix(phone_labels, cluster_labels)
    return float(counts.max(axis=1).sum() / counts.sum())


def mutual_information(phone_labels, cluster_labels):
    '''Compute mutual information between phone and cluster labels.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        Float mutual information value in nats.
    '''
    joint = joint_distribution(phone_labels, cluster_labels)
    phone_marginal = joint.sum(axis=1, keepdims=True)
    cluster_marginal = joint.sum(axis=0, keepdims=True)
    denominator = phone_marginal * cluster_marginal
    mask = joint > 0.0
    values = joint[mask] * np.log(joint[mask] / denominator[mask])
    return float(values.sum())


def marginals(phone_labels, cluster_labels):
    '''Compute p(phone) and p(cluster) from the joint distribution.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.

    returns:        Tuple `(phone_marginal, cluster_marginal)` of 1D NumPy
                    arrays.
    '''
    joint = joint_distribution(phone_labels, cluster_labels)
    return joint.sum(axis=1), joint.sum(axis=0)


def entropy(probabilities):
    '''Compute Shannon entropy from a probability vector or matrix.

    probabilities:  NumPy array or array-like object containing probabilities.
                    Zero-probability entries are ignored.

    returns:        Float entropy value in nats.
    '''
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0: return 0.0
    return float(-np.sum(positive * np.log(positive)))


def filter_valid_frames(phone_labels, cluster_labels, invalid_label=None):
    '''Filter aligned labels and drop invalid frames.

    phone_labels:   1D sequence of frame-level phone labels.
    cluster_labels: 1D sequence of frame-level discrete unit labels.
    invalid_label:  Optional sentinel value to remove from either stream.

    returns:        Tuple `(phone_labels, cluster_labels)` after filtering,
                    both as 1D NumPy object arrays.
    '''
    phone_labels = _as_1d_labels(phone_labels)
    cluster_labels = _as_1d_labels(cluster_labels)

    if phone_labels.size == 0:
        raise ValueError('phone_labels and cluster_labels must not be empty')
    if phone_labels.size != cluster_labels.size:
        raise ValueError(
            'phone_labels and cluster_labels must have the same length')
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


def _count_matrix(phone_labels, cluster_labels):
    phone_labels, cluster_labels = filter_valid_frames(phone_labels,
        cluster_labels)
    phone_inverse, n_phones = _inverse_indices(phone_labels)
    cluster_inverse, n_clusters = _inverse_indices(cluster_labels)
    counts = np.zeros((n_phones, n_clusters), dtype=float)
    np.add.at(counts, (phone_inverse, cluster_inverse), 1.0)
    return counts


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


def _as_1d_labels(labels):
    array = np.asarray(labels, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)


def _is_invalid(value, invalid_label):
    if invalid_label is None:
        return False
    if isinstance(invalid_label, float) and np.isnan(invalid_label):
        return isinstance(value, float) and np.isnan(value)
    return value == invalid_label
