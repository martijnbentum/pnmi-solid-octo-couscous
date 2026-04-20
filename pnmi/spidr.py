import numpy as np

from .metrics import evaluate_labels


def evaluate_streams(phone_labels, codebooks, mode='per_stream',
    layers=None, start_layer=None, end_layer=None, stream_axis=0,
    invalid_label=None, return_diagnostics=False):
    '''Evaluate one or more discrete streams against phone labels.

    This is the main entry point for SpidR-style evaluation where a model
    emits multiple discrete streams or codebooks. It supports three views:
    per-stream evaluation, joint-token evaluation, and pooled summaries
    across streams.

    phone_labels:        1D sequence of frame-level phone labels.
    codebooks:           Discrete streams provided as a dictionary keyed by
                         layer/stream id, a 2D NumPy array, or a list/tuple of
                         streams. Each selected stream must have the same frame
                         length as `phone_labels`.
    mode:                One of `per_stream`, `joint_token`, or
                         `pooled_summary`.
    layers:              Optional explicit list of stream keys to evaluate.
    start_layer:         Optional inclusive lower bound for integer stream keys.
    end_layer:           Optional inclusive upper bound for integer stream keys.
    stream_axis:         For 2D NumPy inputs, axis that indexes streams.
                         `0` means `(streams, frames)`, `1` means
                         `(frames, streams)`.
    invalid_label:       Optional sentinel value removed from the phone stream
                         or any selected codebook stream before scoring.
    return_diagnostics:  If True, include diagnostics from the underlying
                         single-stream evaluations.

    returns:             For `per_stream`, a dictionary keyed by stream id.
                         For `joint_token`, one result dictionary computed on
                         tuple-valued joint labels. For `pooled_summary`, an
                         aggregate summary across streams with mean, min, max,
                         and valid-frame weighted mean.
    '''
    stream_axis = _resolve_stream_axis(phone_labels, codebooks, stream_axis)
    streams = select_codebook_streams(codebooks, layers=layers,
        start_layer=start_layer, end_layer=end_layer, stream_axis=stream_axis)

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


def select_codebook_streams(codebooks, layers=None, start_layer=None,
    end_layer=None, stream_axis=0):
    '''Select one or more codebook streams by key or inclusive layer range.

    codebooks:     Discrete streams as a dict, 2D NumPy array, list, tuple,
                   or single 1D stream.
    layers:        Optional explicit list of stream keys to retain.
    start_layer:   Optional inclusive lower bound for integer stream keys.
    end_layer:     Optional inclusive upper bound for integer stream keys.
    stream_axis:   Axis of the input matrix that indexes streams when
                   `codebooks` is a 2D NumPy array.

    returns:       Dictionary mapping stream keys to 1D NumPy object arrays.
                   All returned streams are guaranteed to have the same length.
    '''
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
    '''Combine one or more streams into tuple-valued frame labels.

    This is useful when several codebooks carry complementary phonetic
    information and you want to score the combined discrete symbol rather than
    each stream independently.

    codebooks:     Discrete streams as a dict, 2D NumPy array, list, tuple,
                   or single 1D stream.
    layers:        Optional explicit list of stream keys to retain.
    start_layer:   Optional inclusive lower bound for integer stream keys.
    end_layer:     Optional inclusive upper bound for integer stream keys.
    stream_axis:   Axis of the input matrix that indexes streams when
                   `codebooks` is a 2D NumPy array.

    returns:       1D NumPy object array where each frame label is a tuple of
                   the selected per-stream values at that frame.
    '''
    streams = select_codebook_streams(codebooks, layers=layers,
        start_layer=start_layer, end_layer=end_layer,
        stream_axis=stream_axis)
    keys = list(streams.keys())
    joint_labels = np.empty(next(iter(streams.values())).shape[0], dtype=object)

    for index in range(joint_labels.size):
        joint_labels[index] = tuple(streams[key][index] for key in keys)
    return joint_labels


def summarize_stream_metrics(per_stream_results, include_diagnostics=False):
    '''Summarise per-stream evaluation results.

    per_stream_results:   Dictionary produced by `evaluate_streams(...,
                          mode='per_stream')`.
    include_diagnostics:  If True, also aggregate MI and entropy diagnostics
                          that are present in the input results.

    returns:              Dictionary with aggregate summaries for PNMI, phone
                          purity, and cluster purity. Each summary contains
                          `mean`, `max`, `min`, and `weighted_mean`. The
                          weighted mean uses each stream's valid-frame count.
    '''
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
            weights),
        'phone_purity': _aggregate_metric(
            [result['phone_purity'] for result in per_stream_results.values()],
            weights),
        'cluster_purity': _aggregate_metric(
            [result['cluster_purity'] for result in per_stream_results.values()],
            weights),
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
                [result[metric_name] for result in per_stream_results.values()],
                weights)

    return summary


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
        return {index: _as_object_array(matrix[index]) for index in
            range(matrix.shape[0])}

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


def _resolve_stream_axis(phone_labels, codebooks, stream_axis):
    if not isinstance(codebooks, np.ndarray) or codebooks.ndim != 2:
        return stream_axis

    if stream_axis not in (0, 1):
        raise ValueError('stream_axis must be 0 or 1')

    frame_count = np.asarray(phone_labels, dtype=object).reshape(-1).size
    matches_axis0 = codebooks.shape[0] == frame_count
    matches_axis1 = codebooks.shape[1] == frame_count

    if matches_axis0 and not matches_axis1:
        return 1
    if matches_axis1 and not matches_axis0:
        return 0
    return stream_axis


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


def _aggregate_metric(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return {
        'mean': float(values.mean()),
        'max': float(values.max()),
        'min': float(values.min()),
        'weighted_mean': float(np.average(values, weights=weights)),
    }


def _as_object_array(values):
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)
