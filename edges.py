import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister, transpile
from qiskit.circuit.library import IntegerComparator
from qiskit_aer import AerSimulator


def encode_fixed_point(x: float, frac_bits: int) -> int:
    """
    Encode a real value into fixed-point integer representation.
    :param x: Real value to be encoded
    :param frac_bits: Number of fractional bits in the fixed-point representation
    :return: Fixed-point encoded integer
    """
    return int(round(x * (2 ** frac_bits)))


def bits_needed(max_value: int) -> int:
    """
    Compute the number of bits needed to represent an integer value.
    :param max_value: Maximum non-negative integer value to be represented
    :return: Minimum number of bits required to represent max_value
    """
    if max_value < 0:
        raise ValueError("max_value must be non negative.")
    return max(1, math.ceil(math.log2(max_value + 1)))


def load_integer_basis(qc: QuantumCircuit, reg, value: int):
    """
    Load an integer value into a quantum register in the computational basis.
    :param qc: Quantum circuit where the basis state will be prepared
    :param reg: Quantum register that will store the basis state
    :param value: Integer value to be loaded into the register
    :return: None
    """
    for k in range(len(reg)):
        if (value >> k) & 1:
            qc.x(reg[k])


def _run(qc: QuantumCircuit, shots: int, backend) -> dict:
    """
    Transpile and execute a quantum circuit.
    :param qc: Quantum circuit to be executed
    :param shots: Number of shots for execution
    :param backend: Qiskit backend used for simulation/execution
    :return: Dictionary with measurement counts
    """
    tqc = transpile(qc, backend, optimization_level=0)
    return backend.run(tqc, shots=shots).result().get_counts()


def compare_single_value_circuit(
    value: float,
    dmax2: float,
    frac_bits: int = 8,
    measure_value: bool = False
) -> QuantumCircuit:
    """
    Create a quantum circuit that compares a value against a squared-distance threshold.
    :param value: Value to be compared
    :param dmax2: Squared-distance threshold used in the comparison
    :param frac_bits: Number of fractional bits in the fixed-point representation
    :param measure_value: Whether to also measure the value register
    :return: Quantum circuit whose flag qubit indicates whether value <= dmax2
    """
    if value < 0 or dmax2 < 0:
        raise ValueError("We need non negative distances")

    value_int = encode_fixed_point(value, frac_bits)
    thr_int = encode_fixed_point(dmax2, frac_bits)

    nbits = bits_needed(max(value_int, thr_int))

    value_reg = QuantumRegister(nbits, name="val")
    flag_reg = QuantumRegister(1, name="flag")

    comp = IntegerComparator(nbits, thr_int + 1, geq=True)

    n_anc = comp.num_qubits - nbits - 1
    anc_reg = AncillaRegister(n_anc, name="anc_cmp") if n_anc > 0 else None

    if measure_value:
        cval = ClassicalRegister(nbits, name="cval")
        cflag = ClassicalRegister(1, name="cflag")

        if anc_reg is not None:
            qc = QuantumCircuit(value_reg, flag_reg, anc_reg, cval, cflag)
        else:
            qc = QuantumCircuit(value_reg, flag_reg, cval, cflag)
    else:
        cflag = ClassicalRegister(1, name="cflag")

        if anc_reg is not None:
            qc = QuantumCircuit(value_reg, flag_reg, anc_reg, cflag)
        else:
            qc = QuantumCircuit(value_reg, flag_reg, cflag)

    load_integer_basis(qc, value_reg, value_int)

    qargs = list(value_reg) + list(flag_reg)
    if anc_reg is not None:
        qargs += list(anc_reg)

    qc.append(comp.to_gate(label=f">={thr_int + 1}"), qargs)

    qc.x(flag_reg[0])

    if measure_value:
        qc.measure(value_reg, cval)
        qc.measure(flag_reg, cflag)
    else:
        qc.measure(flag_reg, cflag)

    return qc


def compare_single_value(
    value: float,
    dmax2: float,
    frac_bits: int = 8,
    backend=None
) -> int:
    """
    Execute the quantum comparison circuit for a single value.
    :param value: Value to be compared
    :param dmax2: Squared-distance threshold used in the comparison
    :param frac_bits: Number of fractional bits in the fixed-point representation
    :param backend: Qiskit backend used for simulation/execution
    :return: 1 if value <= dmax2, 0 otherwise
    """
    if backend is None:
        backend = AerSimulator()

    qc = compare_single_value_circuit(
        value=value,
        dmax2=dmax2,
        frac_bits=frac_bits,
        measure_value=False
    )

    counts = _run(qc, shots=1, backend=backend)

    bitstring = next(iter(counts.keys()))
    return int(bitstring)


def adjacency_matrix_quantum_value_by_value(
    A: np.ndarray,
    dmax2: float,
    frac_bits: int = 8,
    diagonal_zero: bool = True,
    verbose: bool = False,
    backend=None
) -> np.ndarray:
    """
    Build an adjacency matrix from a squared-distance matrix using quantum comparison value by value.
    :param A: Matrix of squared distances
    :param dmax2: Squared-distance threshold used to define adjacency
    :param frac_bits: Number of fractional bits in the fixed-point representation
    :param diagonal_zero: Whether to force the diagonal entries of the adjacency matrix to zero
    :param verbose: Whether to print each comparison result
    :param backend: Qiskit backend used for simulation/execution
    :return: Adjacency matrix with entries equal to 0 or 1
    """
    if backend is None:
        backend = AerSimulator()

    A = np.array(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("Must be a 2D matrix")

    if np.any(A < 0):
        raise ValueError("Square distances must be positive")

    n, m = A.shape
    Adj = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            if diagonal_zero and i == j:
                Adj[i, j] = 0
                continue

            adj_bit = compare_single_value(
                value=A[i, j],
                dmax2=dmax2,
                frac_bits=frac_bits,
                backend=backend
            )

            Adj[i, j] = adj_bit

            if verbose:
                print(f"A[{i},{j}] = {A[i,j]:.6f}  -> Adj[{i},{j}] = {adj_bit}")

    return Adj


def adjacency_matrix_classical(
    A: np.ndarray,
    dmax2: float,
    diagonal_zero: bool = True
) -> np.ndarray:
    """
    Build an adjacency matrix classically from a squared-distance matrix.
    :param A: Matrix of squared distances
    :param dmax2: Squared-distance threshold used to define adjacency
    :param diagonal_zero: Whether to force the diagonal entries of the adjacency matrix to zero
    :return: Classical adjacency matrix with entries equal to 0 or 1
    """
    A = np.array(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("Must be a 2D matrix")

    Adj = (A <= dmax2).astype(int)

    if diagonal_zero:
        np.fill_diagonal(Adj, 0)

    return Adj


if __name__ == "__main__":
    A = np.array([
        [0.00, 1.00, 1.00, 2.00, 4.00, 4.00, 5.00, 5.00],
        [1.00, 0.00, 2.00, 1.00, 1.00, 5.00, 2.00, 4.00],
        [1.00, 2.00, 0.00, 1.00, 5.00, 1.00, 4.00, 2.00],
        [2.00, 1.00, 1.00, 0.00, 2.00, 2.00, 1.00, 1.00],
        [4.00, 1.00, 5.00, 2.00, 0.00, 8.00, 1.00, 5.00],
        [4.00, 5.00, 1.00, 2.00, 8.00, 0.00, 5.00, 1.00],
        [5.00, 2.00, 4.00, 1.00, 1.00, 5.00, 0.00, 2.00],
        [5.00, 4.00, 2.00, 1.00, 5.00, 1.00, 2.00, 0.00],
    ])

    dmax2 = 2
    frac_bits = 8
    backend = AerSimulator()

    Adj_quantum = adjacency_matrix_quantum_value_by_value(
        A=A,
        dmax2=dmax2,
        frac_bits=frac_bits,
        diagonal_zero=True,
        verbose=False,
        backend=backend
    )

    Adj_classical = adjacency_matrix_classical(
        A=A,
        dmax2=dmax2,
        diagonal_zero=True
    )

    print("Squared-distance matrix A:")
    print(A)
    print()

    print(f"Threshold dmax² = {dmax2}")
    print()

    print("Quantum adjacency matrix:")
    print(Adj_quantum)
    print()

    print("Classical adjacency matrix:")
    print(Adj_classical)
    print()

    print("Difference matrix:")
    print(Adj_quantum - Adj_classical)
    print()

    print("Are they equal?")
    print(np.array_equal(Adj_quantum, Adj_classical))