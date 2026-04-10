import numpy as np

from .metrics import evaluate_labels


def _as_object_array(values):
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)


def _normalise_streams(codebooks, stream_axis=0):
    if isinstance(codebooks, dict):
        return {key: _as_object_array(value) for key, value in
            codebooks.items()}

    if isinstance(codebooks, np.ndarray):
        if codebooks.ndim == 1:
            return {0: _as_object_array(codebooks)}
        if codebooks.ndim != 2:
            raise ValueError('codebooks array must be 1D or 2D')
        if stream_axis not in (0, 1):
            raise ValueError('stream_axis must be 0 or 1')
        matrix = codebooks if stream_axis == 0 else codebooks.T
        return {
            index: _as_object_array(matrix[index])
            for index in range(matrix.shape[0])
        }

    if isinstance(codebooks, (list, tuple)):
        if not codebooks:
            raise ValueError('codebooks must not be empty')

        first = np.asarray(codebooks[0], dtype=object)
        if first.ndim == 0:
            return {0: _as_object_array(codebooks)}

        streams = {}
        for index, value in enumerate(codebooks):
            streams[index] = _as_object_array(value)
        return streams

    return {0: _as_object_array(codebooks)}


def _selected_keys(all_keys, layers=None, start_layer=None, end_layer=None):
    if layers is not None and (start_layer is not None or end_layer is not None):
        raise ValueError('use layers or a layer range, not both')

    if layers is not None:
        selected = list(layers)
    elif start_layer is not None or end_layer is not None:
        if any(not isinstance(key, int) for key in all_keys):
            raise ValueError('layer ranges require integer stream keys')
        lower = min(all_keys) if start_layer is None else start_layer
        upper = max(all_keys) if end_layer is None else end_layer
        selected = [key for key in all_keys if lower <= key <= upper]
    else:
        selected = list(all_keys)

    missing = [key for key in selected if key not in all_keys]
    if missing:
        raise ValueError(f'unknown stream keys: {missing}')
    if not selected:
        raise ValueError('no codebook streams selected')
    return selected


def select_codebook_streams(codebooks, layers=None, start_layer=None,
    end_layer=None, stream_axis=0):
    '''Select one or more codebook streams by key or inclusive layer range.'''
    streams = _normalise_streams(codebooks, stream_axis=stream_axis)
    selected = _selected_keys(list(streams.keys()), layers=layers,
        start_layer=start_layer, end_layer=end_layer)

    selected_streams = {key: streams[key] for key in selected}
    lengths = {stream.shape[0] for stream in selected_streams.values()}
    if len(lengths) != 1:
        raise ValueError('all selected codebook streams must have the same length')
    return selected_streams


def build_joint_labels(codebooks, layers=None, start_layer=None,
    end_layer=None, stream_axis=0):
    '''Combine one or more codebook streams into tuple-valued joint labels.'''
    streams = select_codebook_streams(codebooks, layers=layers,
        start_layer=start_layer, end_layer=end_layer,
        stream_axis=stream_axis)
    keys = list(streams.keys())
    joint_labels = np.empty(next(iter(streams.values())).shape[0], dtype=object)

    for index in range(joint_labels.size):
        joint_labels[index] = tuple(streams[key][index] for key in keys)

    return joint_labels


def _aggregate_metric(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return {
        'mean': float(values.mean()),
        'max': float(values.max()),
        'min': float(values.min()),
        'weighted_mean': float(np.average(values, weights = weights)),
    }


def summarize_stream_metrics(per_stream_results, include_diagnostics=False):
    '''Summarise metrics across multiple discrete streams.'''
    if not per_stream_results:
        raise ValueError('per_stream_results must not be empty')

    weights = np.array(
        [result['valid_frame_count'] for result in per_stream_results.values()],
        dtype=float)
    summary = {
        'stream_count': len(per_stream_results),
        'valid_frame_count_total': int(weights.sum()),
        'pnmi': _aggregate_metric(
            [result['pnmi'] for result in per_stream_results.values()],
            weights,
        ),
        'phone_purity': _aggregate_metric(
            [result['phone_purity'] for result in per_stream_results.values()],
            weights,
        ),
        'cluster_purity': _aggregate_metric(
            [result['cluster_purity'] for result in per_stream_results.values()],
            weights,
        ),
    }

    if include_diagnostics:
        for metric_name in [
            'mutual_information',
            'phone_entropy',
            'cluster_entropy',
            'mi_over_phone_entropy',
            'mi_over_cluster_entropy',
        ]:
            summary[metric_name] = _aggregate_metric(
                [
                    result[metric_name] for result in per_stream_results.values()
                ],
                weights,
            )

    return summary


def evaluate_streams(phone_labels, codebooks, mode='per_stream',
    layers=None, start_layer=None, end_layer=None, stream_axis=0,
    invalid_label=None, return_diagnostics=False):
    '''Evaluate one or more discrete streams against frame-level phone labels.

    mode: `per_stream`, `joint_token`, or `pooled_summary`
    '''
    streams = select_codebook_streams(codebooks, layers=layers,
        start_layer=start_layer, end_layer=end_layer,
        stream_axis=stream_axis)

    if mode == 'per_stream':
        return {
            key: evaluate_labels(phone_labels, stream,
                invalid_label=invalid_label,
                return_diagnostics=return_diagnostics)
            for key, stream in streams.items()
        }

    if mode == 'joint_token':
        joint_labels = build_joint_labels(streams, stream_axis=stream_axis)
        result = evaluate_labels(phone_labels, joint_labels,
            invalid_label=invalid_label,
            return_diagnostics=return_diagnostics)
        result['stream_count'] = len(streams)
        return result

    if mode == 'pooled_summary':
        per_stream = evaluate_streams(phone_labels, streams,
            mode='per_stream', invalid_label=invalid_label,
            return_diagnostics=return_diagnostics)
        return summarize_stream_metrics(per_stream,
            include_diagnostics=return_diagnostics)

    raise ValueError(f'unknown evaluation mode: {mode}')
