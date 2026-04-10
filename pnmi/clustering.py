import numpy as np


def cluster_hidden_states(hidden_states, n_clusters, random_state=0,
    **kmeans_kwargs):
    '''Cluster frame-level hidden states and return discrete labels.

    This helper is intended for HuBERT-style or wav2vec2-style workflows
    where you first derive frame-level continuous representations and then
    discretize them before PNMI evaluation. Scikit-learn is imported lazily
    so the main package does not require it unless clustering is used.

    hidden_states:   2D array of shape `(frames, feature_dim)`.
    n_clusters:      Number of KMeans clusters to fit.
    random_state:    Random seed passed to scikit-learn KMeans.
    kmeans_kwargs:   Additional keyword arguments forwarded to KMeans.

    returns:         1D NumPy array of cluster ids with length `frames`.
    '''
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            'cluster_hidden_states requires scikit-learn to be installed'
        ) from exc

    hidden_states = np.asarray(hidden_states, dtype=float)
    if hidden_states.ndim != 2:
        raise ValueError('hidden_states must be a 2D array of shape (frames, dim)')

    model = KMeans(n_clusters=n_clusters, random_state=random_state,
        n_init='auto', **kmeans_kwargs)
    return model.fit_predict(hidden_states)
