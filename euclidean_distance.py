import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator


def _run(qc: QuantumCircuit, shots: int, backend) -> dict:
    """
    Transpile and execute a quantum circuit.
    :param qc: Quantum circuit to be executed
    :param shots: Number of shots for execution
    :param backend: Qiskit backend used for simulation/execution
    :return: Dictionary with measurement counts
    """
    tqc = transpile(qc, backend, optimization_level=2)
    return backend.run(tqc, shots=shots).result().get_counts()


def _pad_and_normalize(v: np.ndarray):
    """
    Normalize a vector and pad it so its dimension becomes a power of 2.
    :param v: Input real or complex vector
    :return: Normalized and padded vector, original norm, number of qubits, and original dimension
    """
    v = np.asarray(v, dtype=complex)
    d = len(v)
    norm_v = np.linalg.norm(v)
    if norm_v <= 1e-12:
        raise ValueError("Zero vector is not permitted")

    nqubits = max(1, math.ceil(math.log2(max(2, d))))
    dim = 2 ** nqubits

    v_normed = v / norm_v
    v_pad = np.zeros(dim, dtype=complex)
    v_pad[:d] = v_normed
    return v_pad, norm_v, nqubits, d


def hadamard_test_circuit(a: np.ndarray, b: np.ndarray):
    """
    Build the Hadamard test circuit to estimate the real part of the inner product between two normalized states.
    :param a: First input vector
    :param b: Second input vector
    :return: Quantum circuit for the Hadamard test, norm of a, and norm of b
    """
    a_pad, norm_a, nq_a, d_a = _pad_and_normalize(a)
    b_pad, norm_b, nq_b, d_b = _pad_and_normalize(b)

    if d_a != d_b:
        raise ValueError("Vectors must have same dimension")
    if nq_a != nq_b:
        raise ValueError("number of qubits must match number of qubits")

    nq = nq_a

    A_circ = QuantumCircuit(nq, name="A")
    A_circ.append(StatePreparation(a_pad, label="|a_hat>"), range(nq))

    B_circ = QuantumCircuit(nq, name="B")
    B_circ.append(StatePreparation(b_pad, label="|b_hat>"), range(nq))

    U_circ = QuantumCircuit(nq, name="U=A†B")
    U_circ.compose(B_circ, inplace=True)
    U_circ.compose(A_circ.inverse(), inplace=True)

    U_gate = U_circ.to_gate(label="A†B")
    cU_gate = U_gate.control(1)

    anc = QuantumRegister(1, "anc")
    data = QuantumRegister(nq, "data")
    qc = QuantumCircuit(anc, data)

    qc.h(anc[0])
    qc.append(cU_gate, [anc[0]] + list(data))
    qc.h(anc[0])

    return qc, norm_a, norm_b


def estimate_inner_product_real_hadamard(
    a: np.ndarray,
    b: np.ndarray,
    shots: int = 8192,
    backend=None,
):
    """
    Estimate the real part of the inner product between two normalized vectors using the Hadamard test.
    :param a: First input vector
    :param b: Second input vector
    :param shots: Number of shots for the quantum execution
    :param backend: Qiskit backend used for simulation/execution
    :return: Estimated real inner product, norm of a, and norm of b
    """
    if backend is None:
        backend = AerSimulator()

    qc, norm_a, norm_b = hadamard_test_circuit(a, b)

    qc_m = qc.copy()
    c = ClassicalRegister(1, "c")
    qc_m.add_register(c)
    qc_m.measure(0, c[0])

    counts = _run(qc_m, shots=shots, backend=backend)
    p0 = counts.get("0", 0) / shots

    inner_product = 2.0 * p0 - 1.0
    inner_product = float(np.clip(inner_product, -1.0, 1.0))

    return inner_product, norm_a, norm_b


def estimate_d2_hadamard(
    a: np.ndarray,
    b: np.ndarray,
    shots: int = 8192,
    backend=None,
):
    """
    Estimate the squared Euclidean distance between two vectors using the Hadamard test.
    :param a: First input vector
    :param b: Second input vector
    :param shots: Number of shots for the quantum execution
    :param backend: Qiskit backend used for simulation/execution
    :return: Estimated squared Euclidean distance between a and b
    """
    inner_product, norm_a, norm_b = estimate_inner_product_real_hadamard(
        a, b, shots=shots, backend=backend
    )

    d2 = norm_a**2 + norm_b**2 - 2.0 * norm_a * norm_b * inner_product
    d2 = max(0.0, float(np.real(d2)))
    return d2


def estimate_distance_matrix_hadamard(
    X: np.ndarray,
    shots: int = 8192,
    backend=None,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Compute the full pairwise squared-distance matrix using the Hadamard-test estimator.
    :param X: Array of shape (N, d), where each row is a point/vector
    :param shots: Number of shots for each quantum estimation
    :param backend: Qiskit backend used for simulation/execution
    :param symmetric: Whether to compute only the upper triangle and mirror it
    :return: Matrix of shape (N, N) with estimated squared distances
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, d)")

    N = X.shape[0]

    if backend is None:
        backend = AerSimulator()

    D = np.zeros((N, N), dtype=float)

    if symmetric:
        for i in range(N):
            D[i, i] = 0.0
            for j in range(i + 1, N):
                d2 = estimate_d2_hadamard(X[i], X[j], shots=shots, backend=backend)
                D[i, j] = d2
                D[j, i] = d2
    else:
        for i in range(N):
            for j in range(N):
                if i == j:
                    D[i, j] = 0.0
                else:
                    D[i, j] = estimate_d2_hadamard(X[i], X[j], shots=shots, backend=backend)

    return D


def exact_distance_matrix_classical(A: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise squared-distance matrix classically.
    :param A: Array of shape (N, d), where each row is a point/vector
    :return: Matrix of shape (N, N) with exact squared distances
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, d)")

    diff = A[:, None, :] - A[None, :, :]
    D = np.sum(diff ** 2, axis=2)
    return D


if __name__ == "__main__":
    X = np.array([
        [1.2, 2.0, 0.7],
        [1.0, 1.5, 1.1],
        [0.5, 2.2, 0.3],
        [1.8, 0.9, 1.4],
    ])

    shots = 20000
    backend = AerSimulator()

    D_est = estimate_distance_matrix_hadamard(X, shots=shots, backend=backend)
    D_exact = exact_distance_matrix_classical(X)

    np.set_printoptions(precision=6, suppress=True)

    print("Estimated squared-distance matrix:")
    print(D_est)
    print()

    print("Exact classical squared-distance matrix:")
    print(D_exact)
    print()

    print("Absolute error matrix:")
    print(np.abs(D_est - D_exact))
    print()

    print("Max absolute error:")
    print(np.max(np.abs(D_est - D_exact)))