import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.datasets import make_blobs, make_moons, make_circles
from qiskit_aer import AerSimulator

from euclidean_distance import (
    estimate_distance_matrix_hadamard,
    exact_distance_matrix_classical,
)

from edges import (
    adjacency_matrix_quantum_value_by_value,
    adjacency_matrix_classical,
)


def choose_threshold_from_distances(
    D: np.ndarray,
    dataset_name: str,
    percentile: float = 35.0,
):
    """
    Choose dmax² from the off-diagonal entries of the distance matrix.
    For small datasets, use a more stable dataset-specific rule.
    :param D: Squared-distance matrix
    :param dataset_name: Dataset type
    :param percentile: Fallback percentile
    :return: dmax²
    """
    D = np.asarray(D, dtype=float)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")

    mask = ~np.eye(D.shape[0], dtype=bool)
    values = D[mask]

    if values.size == 0:
        raise ValueError("Distance matrix must have at least two points.")

    dataset_name = dataset_name.lower()

    if dataset_name == "blobs":
        return float(np.percentile(values, 20.0))
    if dataset_name == "moons":
        return float(np.percentile(values, 10.0))
    if dataset_name == "circles":
        return float(np.percentile(values, 10.0))

    return float(np.percentile(values, percentile))


def generate_dataset(
    dataset_name: str = "moons",
    n_samples: int = 12,
    noise: float = 0.05,
    random_state: int = 42,
):
    """
    Generate a small classical dataset commonly used in spectral clustering.
    :param dataset_name: One of {'blobs', 'moons', 'circles'}
    :param n_samples: Number of points
    :param noise: Noise level for moons/circles
    :param random_state: Random seed
    :return: X, y
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "blobs":
        X, y = make_blobs(
            n_samples=n_samples,
            centers=2,
            cluster_std=0.60,
            n_features=2,
            random_state=random_state,
        )
    elif dataset_name == "moons":
        X, y = make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=random_state,
        )
    elif dataset_name == "circles":
        X, y = make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=random_state,
        )
    else:
        raise ValueError("dataset_name must be 'blobs', 'moons', or 'circles'.")

    return X, y


def count_undirected_edge_differences(A1: np.ndarray, A2: np.ndarray) -> int:
    """
    Count how many undirected edges differ between two adjacency matrices.
    Assumes symmetric adjacency matrices.
    """
    if A1.shape != A2.shape:
        raise ValueError("Adjacency matrices must have the same shape.")

    diff = np.triu(A1 != A2, k=1)
    return int(np.sum(diff))


def adjacency_to_edge_list(A: np.ndarray):
    """
    Convert a symmetric adjacency matrix into a list of undirected edges.
    """
    n = A.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                edges.append((i, j))
    return edges


def build_graph_from_adjacency(A: np.ndarray) -> nx.Graph:
    """
    Create a NetworkX graph from an adjacency matrix.
    """
    G = nx.Graph()
    n = A.shape[0]
    G.add_nodes_from(range(n))
    G.add_edges_from(adjacency_to_edge_list(A))
    return G


def compare_graphs(
    X: np.ndarray,
    y: np.ndarray,
    Adj_classical: np.ndarray,
    Adj_quantum: np.ndarray,
    dataset_name: str,
    dmax2: float,
):
    """
    Plot classical and quantum graphs side by side using the same node positions.
    """
    G_classical = build_graph_from_adjacency(Adj_classical)
    G_quantum = build_graph_from_adjacency(Adj_quantum)

    pos = {i: X[i] for i in range(len(X))}
    node_colors = y

    n_edges_classical = len(G_classical.edges())
    n_edges_quantum = len(G_quantum.edges())
    n_diff_edges = count_undirected_edge_differences(Adj_classical, Adj_quantum)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    nx.draw(
        G_classical,
        pos=pos,
        ax=axes[0],
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Set1,
        node_size=400,
        font_size=10,
    )
    axes[0].set_title(
        f"Classical graph\n"
        f"edges = {n_edges_classical}, dmax² = {dmax2:.4f}"
    )

    nx.draw(
        G_quantum,
        pos=pos,
        ax=axes[1],
        with_labels=True,
        node_color=node_colors,
        cmap=plt.cm.Set1,
        node_size=400,
        font_size=10,
    )
    axes[1].set_title(
        f"Quantum graph\n"
        f"edges = {n_edges_quantum}, differing edges = {n_diff_edges}"
    )

    fig.suptitle(f"Graph comparison for dataset: {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    # ============================================================
    # User parameters
    # ============================================================
    dataset_name = "circles"   # 'blobs', 'moons', or 'circles'
    n_samples = 32           # keep small for quantum simulation cost
    noise = 0.06
    random_state = 42
    shots = 20000
    frac_bits = 4

    backend = AerSimulator()

    # ============================================================
    # 1. Generate dataset
    # ============================================================
    X, y = generate_dataset(
        dataset_name=dataset_name,
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    # ============================================================
    # 2. Compute distance matrices
    # ============================================================
    print("Generating classical squared-distance matrix...")
    D_classical = exact_distance_matrix_classical(X)

    print("Generating quantum squared-distance matrix...")
    D_quantum = estimate_distance_matrix_hadamard(
        X,
        shots=shots,
        backend=backend,
        symmetric=True,
    )

    # ============================================================
    # 3. Choose threshold from classical matrix
    #    (more stable than using noisy quantum estimates)
    # ============================================================
    dmax2 = choose_threshold_from_distances(
        D=D_classical,
        dataset_name=dataset_name,
    )

    # ============================================================
    # 4. Build adjacency matrices
    # ============================================================
    print(f"Chosen threshold dmax² = {dmax2:.6f}")

    Adj_classical = adjacency_matrix_classical(
        A=D_classical,
        dmax2=dmax2,
        diagonal_zero=True,
    )

    Adj_quantum = adjacency_matrix_quantum_value_by_value(
        A=D_quantum,
        dmax2=dmax2,
        frac_bits=frac_bits,
        diagonal_zero=True,
        verbose=False,
        backend=backend,
    )

    # ============================================================
    # 5. Diagnostics
    # ============================================================
    abs_error_D = np.abs(D_quantum - D_classical)
    max_abs_error_D = np.max(abs_error_D)

    edge_diff_count = count_undirected_edge_differences(
        Adj_classical,
        Adj_quantum,
    )

    print("\nDataset points X:")
    print(X)

    print("\nClassical squared-distance matrix:")
    print(D_classical)

    print("\nQuantum squared-distance matrix:")
    print(D_quantum)

    print("\nAbsolute error |D_quantum - D_classical|:")
    print(abs_error_D)

    print(f"\nMax absolute distance error: {max_abs_error_D:.6f}")

    print("\nClassical adjacency matrix:")
    print(Adj_classical)

    print("\nQuantum adjacency matrix:")
    print(Adj_quantum)

    print("\nAdjacency difference matrix (quantum - classical):")
    print(Adj_quantum - Adj_classical)

    print(f"\nNumber of differing undirected edges: {edge_diff_count}")
    print(f"Are adjacency matrices equal? {np.array_equal(Adj_classical, Adj_quantum)}")

    # ============================================================
    # 6. Plot graphs
    # ============================================================
    compare_graphs(
        X=X,
        y=y,
        Adj_classical=Adj_classical,
        Adj_quantum=Adj_quantum,
        dataset_name=dataset_name,
        dmax2=dmax2,
    )


if __name__ == "__main__":
    main()