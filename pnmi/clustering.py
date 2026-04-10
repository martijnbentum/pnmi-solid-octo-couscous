import numpy as np


def cluster_hidden_states(
    hidden_states,
    n_clusters,
    random_state = 0,
    **kmeans_kwargs,
):
    '''Cluster frame-level hidden states and return discrete labels.

    This helper uses scikit-learn when available but is kept optional so the
    core PNMI package remains lightweight.
    '''
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            'cluster_hidden_states requires scikit-learn to be installed'
        ) from exc

    hidden_states = np.asarray(hidden_states, dtype = float)
    if hidden_states.ndim != 2:
        raise ValueError('hidden_states must be a 2D array of shape (frames, dim)')

    model = KMeans(
        n_clusters = n_clusters,
        random_state = random_state,
        n_init = 'auto',
        **kmeans_kwargs,
    )
    return model.fit_predict(hidden_states)
