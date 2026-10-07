import numpy as np
from app.core.types import DistanceMetric

def compute_distance(a: np.ndarray, b: np.ndarray, metric: DistanceMetric) -> float:
    """Calculate the distance between two 1D vectors a and b."""
    if metric == DistanceMetric.EUCLIDEAN:
        return float(np.linalg.norm(a - b))

    elif metric == DistanceMetric.COSINE:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 1.0  # Maximum cosine distance for zero vectors
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clip to [-1.0, 1.0] to prevent floating point inaccuracies
        cos_sim = max(min(cos_sim, 1.0), -1.0)
        return float(1.0 - cos_sim)

    elif metric == DistanceMetric.DOT_PRODUCT:
        # For nearest neighbor, higher dot product means closer.
        # So we define distance as -dot_product.
        return float(-np.dot(a, b))

    raise ValueError(f"Unsupported metric: {metric}")

def batch_compute_distances(query: np.ndarray, matrix: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """
    Vectorized distance calculation between a 1D query vector (D,)
    and a 2D matrix of vectors (N, D).
    Returns an array of shape (N,) containing distances.
    """
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    if metric == DistanceMetric.EUCLIDEAN:
        # ||query - matrix||_2 = sqrt(sum((query - matrix)^2, axis=1))
        diff = matrix - query
        return np.linalg.norm(diff, axis=1)

    elif metric == DistanceMetric.COSINE:
        # Cosine distance = 1 - (A . B) / (||A|| * ||B||)
        norm_q = np.linalg.norm(query)
        norm_matrix = np.linalg.norm(matrix, axis=1)

        # Handle zero vectors safely
        valid_mask = (norm_q > 0) & (norm_matrix > 0)
        dot_products = np.dot(matrix, query)

        similarities = np.zeros(len(matrix), dtype=np.float32)
        if norm_q > 0:
            safe_denom = np.where(valid_mask, norm_matrix * norm_q, 1.0)
            similarities = np.where(valid_mask, dot_products / safe_denom, 0.0)

        # Clip similarities to [-1.0, 1.0]
        similarities = np.clip(similarities, -1.0, 1.0)
        return 1.0 - similarities

    elif metric == DistanceMetric.DOT_PRODUCT:
        # Distance = -dot_product
        return -np.dot(matrix, query)

    raise ValueError(f"Unsupported metric: {metric}")

def distance_to_score(distance: float, metric: DistanceMetric) -> float:
    """Convert distance to human-intuitive similarity score (higher is better)."""
    if metric == DistanceMetric.COSINE:
        # distance in [0, 2], cosine similarity in [-1, 1]
        return float(1.0 - distance)
    elif metric == DistanceMetric.EUCLIDEAN:
        # Euclidean score: 1 / (1 + distance)
        return float(1.0 / (1.0 + max(distance, 0.0)))
    elif metric == DistanceMetric.DOT_PRODUCT:
        # distance was -dot_product, so dot_product = -distance
        return float(-distance)
    return float(-distance)
