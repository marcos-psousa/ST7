import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from qiskit_aer import AerSimulator

from euclidean_distance import (
    estimate_distance_matrix_hadamard,
    exact_distance_matrix_classical,
)

from edges import (
    adjacency_matrix_quantum_value_by_value,
    adjacency_matrix_classical,
)

from incidence_matrix import (
    compute_B_quantum,
    compute_B_classical,
)

from laplacian_matrix import (
    compute_L_quantum,
    compute_L_classical,
)

from spectral_space import (
    solve_lowest_eigenpairs_vqd,
    spectral_embedding_from_eigenvectors,
    run_kmeans_on_embedding,
)

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_matrix_comparison(name: str, Mq: np.ndarray, Mc: np.ndarray):
    print(f"{name} quantum:")
    print(Mq)
    print()
    print(f"{name} classical:")
    print(Mc)
    print()
    print(f"max |{name}q - {name}c|:")
    print(np.max(np.abs(Mq - Mc)))


def solve_lowest_eigenpairs_classical(L: np.ndarray, k: int):
    """
    Compute the k lowest eigenpairs of a symmetric matrix classically.
    :param L: Input symmetric matrix
    :param k: Number of lowest eigenpairs to compute
    :return: Lowest eigenvalues, corresponding eigenvectors, padded matrix, and original dimension
    """
    L = np.asarray(L, dtype=float)

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")

    if not np.allclose(L, L.T, atol=1e-10):
        raise ValueError("L must be symmetric.")

    original_dim = L.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][:k]
    eigenvectors = eigenvectors[:, idx][:, :k]

    return eigenvalues, eigenvectors, L.copy(), original_dim


def generate_dataset(
    dataset_name: str,
    n_points: int,
    noise: float = 0.06,
    n_clusters: int = 2,
    seed: int = 42,
):
    """
    Generate a synthetic dataset for spectral clustering.
    :param dataset_name: One of {'blobs', 'moons', 'circles'}
    :param n_points: Number of samples
    :param noise: Noise level for moons/circles
    :param n_clusters: Number of centers for blobs
    :param seed: Random seed
    :return: X, y_true
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "blobs":
        X, y = make_blobs(
            n_samples=n_points,
            centers=n_clusters,
            cluster_std=1.0,
            random_state=seed,
        )
    elif dataset_name == "moons":
        X, y = make_moons(
            n_samples=n_points,
            noise=noise,
            random_state=seed,
        )
    elif dataset_name == "circles":
        X, y = make_circles(
            n_samples=n_points,
            noise=noise,
            factor=0.5,
            random_state=seed,
        )
    else:
        raise ValueError("dataset_name must be one of {'blobs', 'moons', 'circles'}")

    return X.astype(float), y.astype(int)


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
        return float(np.percentile(values, 25.0))
    if dataset_name == "moons":
        return float(np.percentile(values, 30.0))
    if dataset_name == "circles":
        return float(np.percentile(values, 20.0))

    return float(np.percentile(values, percentile))


def spectral_clustering_pipeline(
    dataset: str = "moons",
    n_points: int = 8,
    n_clusters: int = 2,
    noise: float = 0.06,
    seed: int = 42,
    shots_distance: int = 20000,
    shots_edges: int = 1,
    shots_incidence: int = 50000,
    frac_bits: int = 8,
    eps_B: float = 0.2,
    threshold_percentile: float = 35.0,
    use_AA: bool = False,
    grover_iters=None,
    reps_vqd: int = 2,
    maxiter_vqd: int = 300,
    verbose: bool = True,
):
    """
    Full spectral clustering pipeline:
        X -> D -> A -> B -> L -> eigenvectors -> embedding -> kmeans
    using the functions defined in the 5 external files.

    :return: dictionary with all relevant objects
    """
    _ = shots_edges  # kept for interface compatibility
    backend = AerSimulator()

    # ---------------------------------------------------------
    # 1. Dataset
    # ---------------------------------------------------------
    X, y_true = generate_dataset(
        dataset_name=dataset,
        n_points=n_points,
        noise=noise,
        n_clusters=n_clusters,
        seed=seed,
    )

    if verbose:
        print_section("DATASET")
        print(f"dataset              : {dataset}")
        print(f"n_points             : {n_points}")
        print(f"n_clusters           : {n_clusters}")
        print("X:")
        print(X)
        print()
        print("Ground truth labels:")
        print(y_true)

    # ---------------------------------------------------------
    # 2. Distance matrix
    # ---------------------------------------------------------
    D_quantum = estimate_distance_matrix_hadamard(
        X,
        shots=shots_distance,
        backend=backend,
        symmetric=True,
    )

    D_classical = exact_distance_matrix_classical(X)

    if verbose:
        print_section("DISTANCE MATRICES")
        print_matrix_comparison("D", D_quantum, D_classical)

    # ---------------------------------------------------------
    # 3. Adjacency matrix
    # ---------------------------------------------------------
    mask = ~np.eye(D_quantum.shape[0], dtype=bool)
    scale = np.max(D_quantum[mask])

    if scale <= 1e-12:
        scale = 1.0

    D_quantum_scaled = D_quantum / scale
    D_classical_scaled = D_classical / scale

    dmax2 = choose_threshold_from_distances(
        D_quantum_scaled,
        dataset_name=dataset,
        percentile=threshold_percentile,
    )

    A_quantum = adjacency_matrix_quantum_value_by_value(
        A=D_quantum_scaled,
        dmax2=dmax2,
        frac_bits=frac_bits,
        diagonal_zero=True,
        verbose=False,
        backend=backend,
    )

    A_classical = adjacency_matrix_classical(
        A=D_classical_scaled,
        dmax2=dmax2,
        diagonal_zero=True,
    )

    if verbose:
        print_section("ADJACENCY MATRICES")
        print(f"scale                : {scale:.6f}")
        print(f"threshold dmax²      : {dmax2:.6f}")
        print()
        print("D quantum scaled:")
        print(D_quantum_scaled)
        print()
        print("D classical scaled:")
        print(D_classical_scaled)
        print()
        print_matrix_comparison("A", A_quantum, A_classical)

    # ---------------------------------------------------------
    # 4. Incidence-like matrix B_tilde
    # ---------------------------------------------------------
    B_quantum = compute_B_quantum(
        A=A_quantum,
        eps_B=eps_B,
        shots=shots_incidence,
        use_AA=use_AA,
        grover_iters=grover_iters,
    )

    B_classical = compute_B_classical(
        A=A_classical,
        eps_B=eps_B,
    )

    if verbose:
        print_section("INCIDENCE MATRICES")
        print_matrix_comparison("B", B_quantum, B_classical)

    # ---------------------------------------------------------
    # 5. Laplacian-like matrix L = B B^T
    # ---------------------------------------------------------
    L_quantum = compute_L_quantum(
        B=B_quantum,
        shots=shots_distance,
        backend=backend,
        symmetric=True,
    )

    L_classical = compute_L_classical(B_classical)

    if verbose:
        print_section("LAPLACIAN MATRICES")
        print_matrix_comparison("L", L_quantum, L_classical)

    # ---------------------------------------------------------
    # 6. Lowest eigenpairs via VQD
    # ---------------------------------------------------------
    eigenvalues_q, eigenvectors_q, L_pad_q, original_dim_q = solve_lowest_eigenpairs_vqd(
        L=L_quantum,
        k=n_clusters,
        reps=reps_vqd,
        maxiter=maxiter_vqd,
        seed=seed,
    )

    Y_quantum = spectral_embedding_from_eigenvectors(
        eigenvectors=eigenvectors_q,
        original_dim=original_dim_q,
        k=n_clusters,
    )

    labels_quantum, _ = run_kmeans_on_embedding(
        Y=Y_quantum,
        n_clusters=n_clusters,
        seed=seed,
    )

    if verbose:
        print_section("QUANTUM EIGENPAIRS / EMBEDDING / LABELS")
        print("Quantum eigenvalues:")
        print(eigenvalues_q)
        print()
        print("Quantum embedding Y:")
        print(Y_quantum)
        print()
        print("Quantum labels:")
        print(labels_quantum)

    # ---------------------------------------------------------
    # 7. Classical spectral reference using classical eigendecomposition
    # ---------------------------------------------------------
    eigenvalues_c, eigenvectors_c, L_pad_c, original_dim_c = solve_lowest_eigenpairs_classical(
        L=L_classical,
        k=n_clusters,
    )

    Y_classical = spectral_embedding_from_eigenvectors(
        eigenvectors=eigenvectors_c,
        original_dim=original_dim_c,
        k=n_clusters,
    )

    labels_classical, _ = run_kmeans_on_embedding(
        Y=Y_classical,
        n_clusters=n_clusters,
        seed=seed,
    )

    if verbose:
        print_section("CLASSICAL EIGENPAIRS / EMBEDDING / LABELS")
        print("Classical eigenvalues:")
        print(eigenvalues_c)
        print()
        print("Classical embedding Y:")
        print(Y_classical)
        print()
        print("Classical labels:")
        print(labels_classical)

    # ---------------------------------------------------------
    # 8. Metrics
    # ---------------------------------------------------------
    ari_quantum = adjusted_rand_score(y_true, labels_quantum)
    nmi_quantum = normalized_mutual_info_score(y_true, labels_quantum)

    ari_classical = adjusted_rand_score(y_true, labels_classical)
    nmi_classical = normalized_mutual_info_score(y_true, labels_classical)

    if verbose:
        print_section("FINAL METRICS")
        print(f"ARI quantum          : {ari_quantum:.6f}")
        print(f"NMI quantum          : {nmi_quantum:.6f}")
        print(f"ARI classical        : {ari_classical:.6f}")
        print(f"NMI classical        : {nmi_classical:.6f}")

    return {
        "X": X,
        "y_true": y_true,
        "D_quantum": D_quantum,
        "D_classical": D_classical,
        "D_quantum_scaled": D_quantum_scaled,
        "D_classical_scaled": D_classical_scaled,
        "scale": scale,
        "dmax2": dmax2,
        "A_quantum": A_quantum,
        "A_classical": A_classical,
        "B_quantum": B_quantum,
        "B_classical": B_classical,
        "L_quantum": L_quantum,
        "L_classical": L_classical,
        "eigenvalues_quantum": eigenvalues_q,
        "eigenvalues_classical": eigenvalues_c,
        "Y_quantum": Y_quantum,
        "Y_classical": Y_classical,
        "labels_quantum": labels_quantum,
        "labels_classical": labels_classical,
        "ari_quantum": ari_quantum,
        "nmi_quantum": nmi_quantum,
        "ari_classical": ari_classical,
        "nmi_classical": nmi_classical,
    }


def plot_results(X, y_true, labels_quantum, labels_classical, dataset_name: str):
    """
    Plot ground-truth and predicted clusterings.
    """
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true)
    plt.title(f"Ground truth ({dataset_name})")

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels_quantum)
    plt.title("Quantum spectral clustering")

    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=labels_classical)
    plt.title("Classical spectral reference")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Quantum-inspired spectral clustering pipeline")
    parser.add_argument("--dataset", type=str, default="moons", choices=["blobs", "moons", "circles"])
    parser.add_argument("--n_points", type=int, default=8)
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--shots_distance", type=int, default=50000)
    parser.add_argument("--shots_incidence", type=int, default=50000)
    parser.add_argument("--frac_bits", type=int, default=8)
    parser.add_argument("--eps_B", type=float, default=0.01)
    parser.add_argument("--threshold_percentile", type=float, default=35.0)

    parser.add_argument("--use_AA", action="store_true")
    parser.add_argument("--reps_vqd", type=int, default=2)
    parser.add_argument("--maxiter_vqd", type=int, default=300)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)

    results = spectral_clustering_pipeline(
        dataset=args.dataset,
        n_points=args.n_points,
        n_clusters=args.n_clusters,
        noise=args.noise,
        seed=args.seed,
        shots_distance=args.shots_distance,
        shots_edges=1,
        shots_incidence=args.shots_incidence,
        frac_bits=args.frac_bits,
        eps_B=args.eps_B,
        threshold_percentile=args.threshold_percentile,
        use_AA=args.use_AA,
        grover_iters=None,
        reps_vqd=args.reps_vqd,
        maxiter_vqd=args.maxiter_vqd,
        verbose=True,
    )

    if args.plot:
        plot_results(
            X=results["X"],
            y_true=results["y_true"],
            labels_quantum=results["labels_quantum"],
            labels_classical=results["labels_classical"],
            dataset_name=args.dataset,
        )


if __name__ == "__main__":
    main()