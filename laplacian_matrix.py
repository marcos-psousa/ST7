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
    Normalize a real or complex vector and pad it so its dimension becomes a power of 2.
    :param v: Input real or complex vector
    :return: Normalized and padded vector, original norm, number of qubits, and original dimension
    """
    v = np.asarray(v, dtype=complex)
    d = len(v)
    norm_v = np.linalg.norm(v)

    if norm_v <= 1e-12:
        raise ValueError("Zero vector is not permitted.")

    nq = max(1, math.ceil(math.log2(max(2, d))))
    dim = 2 ** nq

    v_normed = v / norm_v
    v_pad = np.zeros(dim, dtype=complex)
    v_pad[:d] = v_normed
    return v_pad, norm_v, nq, d


def hadamard_overlap_circuit(a: np.ndarray, b: np.ndarray):
    """
    Build the Hadamard test circuit to estimate Re(<â|b̂>).
    :param a: First input vector
    :param b: Second input vector
    :return: Quantum circuit implementing the Hadamard test
    """
    a_pad, norm_a, nq_a, d_a = _pad_and_normalize(a)
    b_pad, norm_b, nq_b, d_b = _pad_and_normalize(b)

    if d_a != d_b:
        raise ValueError("Vectors must have the same original dimension.")

    if nq_a != nq_b:
        raise ValueError("Internal error: incompatible number of qubits.")

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

    return qc


def estimate_inner_product_real_hadamard(
    a: np.ndarray,
    b: np.ndarray,
    shots: int = 8192,
    backend=None,
):
    """
    Estimate Re(<â|b̂>) using the Hadamard test.
    :param a: First input vector
    :param b: Second input vector
    :param shots: Number of shots for the quantum execution
    :param backend: Qiskit backend used for simulation/execution
    :return: Estimated real part of the overlap between the normalized vectors
    """
    if backend is None:
        backend = AerSimulator()

    qc = hadamard_overlap_circuit(a, b)

    qc_m = qc.copy()
    c = ClassicalRegister(1, "c")
    qc_m.add_register(c)
    qc_m.measure(0, c[0])

    counts = _run(qc_m, shots=shots, backend=backend)
    p0 = counts.get("0", 0) / shots

    inner_product = 2.0 * p0 - 1.0
    inner_product = float(np.clip(inner_product, -1.0, 1.0))

    return inner_product


def compute_L_quantum(
    B: np.ndarray,
    shots: int = 8192,
    backend=None,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Compute the full matrix L quantumly, where L[i, j] = Re(<B_i | B_j>).
    If symmetric=True, compute only the upper triangle and mirror it.
    :param B: Matrix whose rows are the vectors B_i
    :param shots: Number of shots for each Hadamard-test estimation
    :param backend: Qiskit backend used for simulation/execution
    :param symmetric: Whether to force symmetry by computing only half the matrix
    :return: Quantum-estimated matrix L
    """
    B = np.asarray(B, dtype=float)

    if B.ndim != 2:
        raise ValueError("B must be a 2D array.")

    if backend is None:
        backend = AerSimulator()

    m = B.shape[0]
    L_quantum = np.zeros((m, m), dtype=float)

    if symmetric:
        for i in range(m):
            L_quantum[i, i] = estimate_inner_product_real_hadamard(
                B[i], B[i], shots=shots, backend=backend
            )
            for j in range(i + 1, m):
                val = estimate_inner_product_real_hadamard(
                    B[i], B[j], shots=shots, backend=backend
                )
                L_quantum[i, j] = val
                L_quantum[j, i] = val
    else:
        for i in range(m):
            for j in range(m):
                L_quantum[i, j] = estimate_inner_product_real_hadamard(
                    B[i], B[j], shots=shots, backend=backend
                )

    return L_quantum


def compute_L_classical(B: np.ndarray) -> np.ndarray:
    """
    Compute the full matrix L classically as B @ B^T.
    :param B: Matrix whose rows are the vectors B_i
    :return: Classical matrix L
    """
    B = np.asarray(B, dtype=float)

    if B.ndim != 2:
        raise ValueError("B must be a 2D array.")

    return B @ B.T


if __name__ == "__main__":
    backend = AerSimulator()
    shots = 20000

    B = np.array([
        [1.0,                 0.0, 0.0, 0.0,               0.0,              0.0],
        [-1.0 / math.sqrt(3), 0.0, 0.0, 1 / math.sqrt(3),  1 / math.sqrt(3), 0.0],
        [0.0,                 0.0, 0.0, -1 / math.sqrt(2), 0.0,              1 / math.sqrt(2)],
        [0.0,                 0.0, 0.0, 0.0,               -1 / math.sqrt(2), -1 / math.sqrt(2)],
    ], dtype=float)

    L_quantum = compute_L_quantum(
        B=B,
        shots=shots,
        backend=backend
    )

    L_classical = compute_L_classical(B)

    print("Matrix B:")
    print(B)

    print("\nQuantum-estimated matrix L:")
    print(L_quantum)

    print("\nClassical matrix L = B @ B^T:")
    print(L_classical)

    print("\nAbsolute error:")
    print(np.abs(L_quantum - L_classical))

    print("\nMax absolute error:")
    print(np.max(np.abs(L_quantum - L_classical)))