"""
Theorem 8  —  A -> |B_tilde_i>
(Kerenidis & Landman, PhysRevA 103, 042415, 2021, Section IV.B)

Mapping:
    |i>|0>  ->  |i>|B_tilde_i>

    B_tilde_{i,(p,q)} = +a_{pq}   if i=p      Eq.(4)
    B_tilde_{i,(p,q)} = -a_{pq}   if i=q      Eq.(4)
    B_tilde_{i,(p,q)} = eps_B     if i not in {p,q}   Section II.A

Circuit steps (Eqs. 6-10, Theorem 8):
    1. EdgeSuperposition  ->  1/sqrt(M) sum_{p<q} |p,q>        Eq.(6)
    2. Equality flags     ->  fp <- [i=p],  fq <- [i=q]        Eq.(7)
    3. Adjacency oracle   ->  adj <- A[p][q]                    QRAM
    4. Value encoding     ->  v1 <- incident, ve <- eps_B       Eq.(8)
    5. Conditional rot.   ->  B_tilde_{i,(p,q)} -> ampl(anc)   Thm.2
    6. Sign encoding      ->  phase -1 when i=q                 Eq.(10)
    7. Uncompute          ->  clears fp, fq, adj, ve, v1
    8. Amplitude Amplif.  ->  amplifies anc=1 branch            Thm.1

Amplitude Amplification -- Section 5.4 of lecture notes:
    |psi> = U_Bi|0>_work      initial state (i_reg fixed)
    O   |x> = (-1)^f(x)|x>   phase oracle (f=1 iff anc=1)
    U0_perp |0>=|0>, |x!=0>=-|x>    reflection about |0> (Sec. 5.4.4)
    U_perp_psi = U_Bi . U0_perp . U_Bi_dag  = 2|psi><psi|-I  (Sec. 5.4.8)
    One Grover iteration:  O . U_perp_psi     (Sec. 5.4.7)
    Optimal iterations:    r ~ pi/4 * sqrt(M/k)   (Sec. 5.4.10)

Simulation: qiskit.quantum_info.Statevector -- exact, no transpiler.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation


# ── utilities ────────────────────────────────────────────────────────

def bits_needed(n):
    """
    Compute the number of bits needed to represent indices from 0 to n - 1.
    :param n: Number of items to be indexed
    :return: Minimum number of bits required
    """
    return max(1, math.ceil(math.log2(max(2, n))))


def all_pairs(n):
    """
    Generate all ordered pairs (p, q) such that p < q.
    :param n: Number of vertices
    :return: List of all pairs (p, q) with p < q
    """
    return [(p, q) for p in range(n) for q in range(p + 1, n)]


def _set_bits(qc, reg, value):
    """
    Prepare a computational basis state |value> in a register using X gates.
    :param qc: Quantum circuit where the basis state will be prepared
    :param reg: Quantum register that will store the basis state
    :param value: Integer value to be loaded into the register
    :return: None
    """
    for k, bit in enumerate(format(value, f"0{len(reg)}b")[::-1]):
        if bit == "1":
            qc.x(reg[k])


def _match_controls(qc, reg, value):
    """
    Apply X gates on zero-bits of a value to create positive controls for MCX.
    :param qc: Quantum circuit where the controls will be prepared
    :param reg: Quantum register used as control register
    :param value: Integer value to be matched
    :return: None
    """
    for k, bit in enumerate(format(value, f"0{len(reg)}b")[::-1]):
        if bit == "0":
            qc.x(reg[k])


# ── Step 1: EdgeSuperposition  Eq.(6) ────────────────────────────────

def step1_edge_superposition(n):
    """
    Build the state preparation gate for the uniform superposition over all pairs (p, q) with p < q.
    :param n: Number of vertices
    :return: StatePreparation gate encoding the edge superposition
    """
    b = bits_needed(n)
    pairs = all_pairs(n)
    M = len(pairs)
    amp = np.zeros(2 ** (2 * b), dtype=complex)

    for p, q in pairs:
        amp[p + (q << b)] = 1.0 / math.sqrt(M)

    return StatePreparation(amp, label="EdgeSup")


# ── Step 2: Equality flags  Eq.(7) ───────────────────────────────────

def _equal_flag(qc, reg_a, reg_b, flag, work):
    """
    Set a flag qubit to 1 iff two registers are equal, using XOR logic and MCX.
    :param qc: Quantum circuit where the comparison is performed
    :param reg_a: First quantum register
    :param reg_b: Second quantum register
    :param flag: Qubit that stores the equality result
    :param work: Workspace register used during the comparison
    :return: None
    """
    b = len(reg_a)

    for k in range(b):
        qc.cx(reg_a[k], work[k])
        qc.cx(reg_b[k], work[k])
        qc.x(work[k])

    qc.mcx(list(work), flag)

    for k in reversed(range(b)):
        qc.x(work[k])
        qc.cx(reg_b[k], work[k])
        qc.cx(reg_a[k], work[k])


def step2_equality_flags(qc, i_reg, p_reg, q_reg, fp, fq, work):
    """
    Compute the equality flags fp = [i = p] and fq = [i = q].
    :param qc: Quantum circuit where the flags will be computed
    :param i_reg: Register holding the fixed vertex index i
    :param p_reg: Register holding the first endpoint p
    :param q_reg: Register holding the second endpoint q
    :param fp: Flag qubit storing whether i = p
    :param fq: Flag qubit storing whether i = q
    :param work: Workspace register used for equality checks
    :return: None
    """
    _equal_flag(qc, i_reg, p_reg, fp, work)
    _equal_flag(qc, i_reg, q_reg, fq, work)


# ── Step 3: Adjacency oracle  (QRAM) ─────────────────────────────────

def step3_adjacency_oracle(qc, p_reg, q_reg, adj, A):
    """
    Load the adjacency value A[p][q] into the adj qubit for pairs with p < q.
    :param qc: Quantum circuit where the oracle is implemented
    :param p_reg: Register holding the first endpoint p
    :param q_reg: Register holding the second endpoint q
    :param adj: Target qubit storing the adjacency value
    :param A: Adjacency matrix
    :return: None
    """
    n = len(A)
    controls = list(p_reg) + list(q_reg)

    for p in range(n):
        for q in range(p + 1, n):
            if A[p][q] == 1:
                _match_controls(qc, p_reg, p)
                _match_controls(qc, q_reg, q)
                qc.mcx(controls, adj)
                _match_controls(qc, q_reg, q)
                _match_controls(qc, p_reg, p)


# ── Step 4: Value encoding  Eq.(8) ───────────────────────────────────

def step4_value_encoding(qc, fp, fq, adj, ve, v1):
    """
    Encode the incident-edge branch and the eps_B branch into auxiliary qubits.
    :param qc: Quantum circuit where the encoding is performed
    :param fp: Flag qubit storing whether i = p
    :param fq: Flag qubit storing whether i = q
    :param adj: Qubit storing the adjacency value A[p][q]
    :param ve: Qubit storing the eps_B branch indicator
    :param v1: Qubit storing the incident branch indicator
    :return: None
    """
    qc.ccx(fp, adj, v1)
    qc.ccx(fq, adj, v1)
    qc.x(fp)
    qc.x(fq)
    qc.ccx(fp, fq, ve)
    qc.x(fq)
    qc.x(fp)


def step4_value_uncompute(qc, fp, fq, adj, ve, v1):
    """
    Uncompute the auxiliary qubits introduced during value encoding.
    :param qc: Quantum circuit where the uncomputation is performed
    :param fp: Flag qubit storing whether i = p
    :param fq: Flag qubit storing whether i = q
    :param adj: Qubit storing the adjacency value A[p][q]
    :param ve: Qubit storing the eps_B branch indicator
    :param v1: Qubit storing the incident branch indicator
    :return: None
    """
    qc.x(fp)
    qc.x(fq)
    qc.ccx(fp, fq, ve)
    qc.x(fq)
    qc.x(fp)
    qc.ccx(fq, adj, v1)
    qc.ccx(fp, adj, v1)


# ── Step 5: Conditional rotation  Theorem 2 ──────────────────────────

def step5_conditional_rotation(qc, ve, v1, anc, eps_B):
    """
    Encode the value of B_tilde_{i,(p,q)} into the amplitude of the ancilla qubit.
    :param qc: Quantum circuit where the conditional rotation is applied
    :param ve: Qubit storing the eps_B branch indicator
    :param v1: Qubit storing the incident branch indicator
    :param anc: Ancilla qubit receiving the amplitude encoding
    :param eps_B: Value used for non-incident entries
    :return: None
    """
    if eps_B > 0:
        qc.cry(2 * math.asin(eps_B), ve, anc)

    qc.cx(v1, anc)


# ── Step 6: Sign encoding  Eq.(10) ───────────────────────────────────

def step6_sign_encoding(qc, fq, v1, anc):
    """
    Encode the negative sign when i = q on an existing edge.
    :param qc: Quantum circuit where the sign encoding is applied
    :param fq: Flag qubit storing whether i = q
    :param v1: Qubit storing the incident branch indicator
    :param anc: Ancilla qubit carrying the amplitude information
    :return: None
    """
    qc.ccz(fq, v1, anc)


# ── Step 7: Uncompute ─────────────────────────────────────────────────

def step7_uncompute(qc, i_reg, p_reg, q_reg, fp, fq, work, adj, ve, v1, A):
    """
    Reverse the operations of steps 4, 3, and 2, leaving only the ancilla encoded.
    :param qc: Quantum circuit where the uncomputation is performed
    :param i_reg: Register holding the fixed vertex index i
    :param p_reg: Register holding the first endpoint p
    :param q_reg: Register holding the second endpoint q
    :param fp: Flag qubit storing whether i = p
    :param fq: Flag qubit storing whether i = q
    :param work: Workspace register used during equality checks
    :param adj: Qubit storing the adjacency value A[p][q]
    :param ve: Qubit storing the eps_B branch indicator
    :param v1: Qubit storing the incident branch indicator
    :param A: Adjacency matrix
    :return: None
    """
    step4_value_uncompute(qc, fp, fq, adj, ve, v1)
    step3_adjacency_oracle(qc, p_reg, q_reg, adj, A)
    step2_equality_flags(qc, i_reg, p_reg, q_reg, fp, fq, work)


# ── Step 8: Amplitude Amplification  Theorem 1 / Sec. 5.4 ────────────

def _U0_perp(qc, qubits):
    """
    Apply the reflection operator U0_perp = 2|0><0| - I on a list of qubits.
    :param qc: Quantum circuit where the reflection is applied
    :param qubits: List of qubits defining the reflection subspace
    :return: None
    """
    n = len(qubits)

    if n == 0:
        return

    if n == 1:
        qc.z(qubits[0])
        return

    for q in qubits:
        qc.x(q)

    qc.h(qubits[-1])
    qc.mcx(list(qubits[:-1]), qubits[-1])
    qc.h(qubits[-1])

    for q in qubits:
        qc.x(q)


def grover_iters_optimal(P_success):
    """
    Compute the optimal number of Grover iterations from the initial success probability.
    :param P_success: Success probability P(anc = 1) before amplitude amplification
    :return: Estimated optimal number of Grover iterations
    """
    if P_success <= 0 or P_success >= 1:
        return 0

    return max(0, round(math.pi / (4 * math.asin(math.sqrt(P_success))) - 0.5))


# ── Full circuit ──────────────────────────────────────────────────────

def build_Bi_circuit(A, i, eps_B=0.2, use_AA=True, grover_iters=None):
    """
    Build the circuit implementing the map |i>|0> -> |B_tilde_i>.
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde will be encoded
    :param eps_B: Value used for non-incident entries
    :param use_AA: Whether to apply amplitude amplification
    :param grover_iters: Number of Grover iterations; if None, use the estimated optimal value
    :return: Quantum circuit implementing the encoding of row i
    """
    A = np.array(A, dtype=int)
    n = len(A)
    b = bits_needed(n)
    M = n * (n - 1) // 2

    i_reg = QuantumRegister(b, "i")
    p_reg = QuantumRegister(b, "p")
    q_reg = QuantumRegister(b, "q")
    fp = QuantumRegister(1, "fp")
    fq = QuantumRegister(1, "fq")
    work = QuantumRegister(b, "w")
    adj = QuantumRegister(1, "adj")
    ve = QuantumRegister(1, "ve")
    v1 = QuantumRegister(1, "v1")
    anc = QuantumRegister(1, "anc")
    regs = [i_reg, p_reg, q_reg, fp, fq, work, adj, ve, v1, anc]

    def _prep_body(qc):
        """
        Apply steps 1 through 7 of the construction, excluding the preparation of i_reg.
        :param qc: Quantum circuit where the preparation body will be applied
        :return: None
        """
        qc.append(step1_edge_superposition(n), list(p_reg) + list(q_reg))
        step2_equality_flags(qc, i_reg, p_reg, q_reg, fp[0], fq[0], work)
        step3_adjacency_oracle(qc, p_reg, q_reg, adj[0], A)
        step4_value_encoding(qc, fp[0], fq[0], adj[0], ve[0], v1[0])
        step5_conditional_rotation(qc, ve[0], v1[0], anc[0], eps_B)
        step6_sign_encoding(qc, fq[0], v1[0], anc[0])
        step7_uncompute(
            qc,
            i_reg,
            p_reg,
            q_reg,
            fp[0],
            fq[0],
            work,
            adj[0],
            ve[0],
            v1[0],
            A,
        )

    qc = QuantumCircuit(*regs)
    _set_bits(qc, i_reg, i)

    if not use_AA:
        _prep_body(qc)
        return qc

    prep_circ = QuantumCircuit(*regs)
    _prep_body(prep_circ)
    U_Bi = prep_circ.to_gate(label="U_Bi")
    U_Bi_dag = U_Bi.inverse()

    if grover_iters is None:
        pairs = all_pairs(n)
        amps = []

        for p, q in pairs:
            fp_ = 1 if i == p else 0
            fq_ = 1 if i == q else 0
            a = int(A[p][q])
            v1_ = 1 if (fp_ or fq_) and a else 0
            ve_ = 1 if not fp_ and not fq_ else 0
            raw = (-1 if fq_ else 1) if v1_ else (eps_B if ve_ else 0.0)
            amps.append(raw / math.sqrt(M))

        P = sum(x ** 2 for x in amps)
        grover_iters = grover_iters_optimal(P)

    qc.append(U_Bi, qc.qubits)

    search_qubits = list(p_reg) + list(q_reg) + [anc[0]]

    for _ in range(grover_iters):
        qc.z(anc[0])
        qc.append(U_Bi_dag, qc.qubits)
        _U0_perp(qc, search_qubits)
        qc.append(U_Bi, qc.qubits)

    return qc


# ── Extraction via Statevector ────────────────────────────────────────

def extract_Bi(qc, A, i, eps_B):
    """
    Extract row i of B_tilde from the anc = 1 branch of the final statevector.
    :param qc: Quantum circuit whose statevector will be analyzed
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being extracted
    :param eps_B: Value used for non-incident entries
    :return: Reconstructed normalized row of B_tilde
    """
    A = np.array(A, dtype=int)
    n = A.shape[0]
    b = bits_needed(n)
    pairs = all_pairs(n)
    N = len(qc.qubits)

    sv = Statevector(qc).data

    off_p = b
    off_q = 2 * b
    off_anc = N - 1

    mask_b = (1 << b) - 1
    amp = {}

    for idx in range(2 ** N):
        if (idx >> off_anc) & 1 != 1:
            continue
        if (idx & mask_b) != i:
            continue

        p_val = (idx >> off_p) & mask_b
        q_val = (idx >> off_q) & mask_b
        amp[(p_val, q_val)] = amp.get((p_val, q_val), 0j) + sv[idx]

    row = np.array([amp.get((p, q), 0j) for p, q in pairs], dtype=complex).real.copy()

    nrm = np.linalg.norm(row)
    if nrm > 1e-12:
        row /= nrm

    row = np.abs(row)

    for col, (p, q) in enumerate(pairs):
        if i == q and A[p, q] == 1:
            row[col] *= -1

    return row


# ── Classical reference ───────────────────────────────────────────────

def classical_Bi(A, i, eps_B=0.2):
    """
    Compute row i of B_tilde classically.
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being computed
    :param eps_B: Value used for non-incident entries
    :return: Classically computed normalized row of B_tilde
    """
    A = np.array(A, dtype=int)
    n = A.shape[0]
    pairs = all_pairs(n)

    row = np.array([
        float(A[p, q]) if i == p else
        -float(A[p, q]) if i == q else
        eps_B
        for p, q in pairs
    ])

    nrm = np.linalg.norm(row)
    return row / nrm if nrm > 1e-12 else row


# ── Public API ────────────────────────────────────────────────────────

def compute_row_B(A, i, eps_B=0.2, use_AA=False, grover_iters=None):
    """
    Compute row i of B_tilde with the quantum circuit and compare it to the classical reference.
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being computed
    :param eps_B: Value used for non-incident entries
    :param use_AA: Whether to apply amplitude amplification
    :param grover_iters: Number of Grover iterations; if None, use the estimated optimal value
    :return: Quantum row, classical row, and maximum absolute error
    """
    A = np.array(A, dtype=int)
    qc = build_Bi_circuit(A, i, eps_B=eps_B, use_AA=use_AA, grover_iters=grover_iters)
    B_q = extract_Bi(qc, A, i, eps_B)
    ref = classical_Bi(A, i, eps_B)
    return B_q, ref, float(np.max(np.abs(B_q - ref)))


def build_Bi_measurement_circuit(A, i, eps_B=0.2, use_AA=True, grover_iters=None):
    """
    Build the measurement circuit for row i of B_tilde, measuring anc, p_reg, q_reg, and i_reg.
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being computed
    :param eps_B: Value used for non-incident entries
    :param use_AA: Whether to apply amplitude amplification
    :param grover_iters: Number of Grover iterations; if None, use the estimated optimal value
    :return: Quantum circuit with measurements added
    """
    qc = build_Bi_circuit(A, i, eps_B=eps_B, use_AA=use_AA, grover_iters=grover_iters)

    n = len(A)
    b = bits_needed(n)

    c = ClassicalRegister(1 + 3 * b, "c")
    qc.add_register(c)

    anc_qubit = qc.qubits[-1]

    i_qubits = qc.qubits[0:b]
    p_qubits = qc.qubits[b:2 * b]
    q_qubits = qc.qubits[2 * b:3 * b]

    qc.measure(anc_qubit, c[0])

    for k, q in enumerate(p_qubits):
        qc.measure(q, c[1 + k])

    for k, q in enumerate(q_qubits):
        qc.measure(q, c[1 + b + k])

    for k, q in enumerate(i_qubits):
        qc.measure(q, c[1 + 2 * b + k])

    return qc


def _decode_counts_key(bitstring, b):
    """
    Decode a Qiskit counts key into anc, p, q, and i values.
    :param bitstring: Measurement bitstring returned by Qiskit
    :param b: Number of bits used for each index register
    :return: Decoded values (anc, p, q, i_val)
    """
    bits = bitstring.replace(" ", "")[::-1]

    anc = int(bits[0])

    p = 0
    for k in range(b):
        p |= (int(bits[1 + k]) << k)

    q = 0
    for k in range(b):
        q |= (int(bits[1 + b + k]) << k)

    i_val = 0
    for k in range(b):
        i_val |= (int(bits[1 + 2 * b + k]) << k)

    return anc, p, q, i_val


def extract_Bi_from_counts(counts, A, i, eps_B=0.2):
    """
    Reconstruct row i of B_tilde from measurement counts by post-selecting anc = 1 and i_reg = i.
    :param counts: Dictionary of measurement counts
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being reconstructed
    :param eps_B: Value used for non-incident entries
    :return: Reconstructed normalized row of B_tilde
    """
    A = np.array(A, dtype=int)
    n = len(A)
    b = bits_needed(n)
    pairs = all_pairs(n)

    edge_counts = {pair: 0 for pair in pairs}
    total_good = 0

    for bitstring, cnt in counts.items():
        anc, p, q, i_val = _decode_counts_key(bitstring, b)

        if anc != 1:
            continue
        if i_val != i:
            continue
        if not (0 <= p < n and 0 <= q < n):
            continue
        if p >= q:
            continue
        if (p, q) not in edge_counts:
            continue

        edge_counts[(p, q)] += cnt
        total_good += cnt

    row = np.zeros(len(pairs), dtype=float)

    if total_good == 0:
        return row

    for col, (p, q) in enumerate(pairs):
        prob_cond = edge_counts[(p, q)] / total_good
        amp_mag = math.sqrt(prob_cond)

        sign = 1.0
        if i == q and A[p, q] == 1:
            sign = -1.0

        row[col] = sign * amp_mag

    nrm = np.linalg.norm(row)
    if nrm > 1e-12:
        row /= nrm

    return row


def compute_row_B_shots(A, i, eps_B=0.2, shots=20000, use_AA=False, grover_iters=None,
                        optimization_level=0, seed_simulator=1234):
    """
    Compute row i of B_tilde from shot-based simulation and compare it to the classical reference.
    :param A: Adjacency matrix
    :param i: Vertex index whose row of B_tilde is being computed
    :param eps_B: Value used for non-incident entries
    :param shots: Number of shots used in the simulation
    :param use_AA: Whether to apply amplitude amplification
    :param grover_iters: Number of Grover iterations; if None, use the estimated optimal value
    :param optimization_level: Transpiler optimization level
    :param seed_simulator: Random seed for the simulator
    :return: Quantum row, classical row, maximum absolute error, counts, and circuit
    """
    backend = AerSimulator(method="statevector")

    qc = build_Bi_measurement_circuit(
        A=A,
        i=i,
        eps_B=eps_B,
        use_AA=use_AA,
        grover_iters=grover_iters,
    )

    tqc = transpile(qc, backend, optimization_level=optimization_level)

    result = backend.run(
        tqc,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()

    counts = result.get_counts()
    B_q = extract_Bi_from_counts(counts, A, i, eps_B=eps_B)
    ref = classical_Bi(A, i, eps_B=eps_B)
    err = float(np.max(np.abs(B_q - ref)))

    return B_q, ref, err, counts, qc


if __name__ == "__main__":
    A = np.array([
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ], dtype=int)

    n = len(A)
    eps_B = 0.2
    shots = 50000

    print(f"{'i':>3}  {'erro':>10}  {'B_quantum(shots)':>25}")
    print("-" * 90)

    for i in range(n):
        B_q, ref, err, counts, qc = compute_row_B_shots(
            A, i, eps_B=eps_B, shots=shots, use_AA=False
        )
        print(f"{i:>3}  {err:>10.6f}  {np.round(B_q, 4)}")

    print("\nReferência clássica:")
    for i in range(n):
        print(f"{i}: {np.round(classical_Bi(A, i, eps_B), 4)}")