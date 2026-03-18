import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# ============================================================
# 1. UTILITÁRIOS
# ============================================================

def bits_needed(n: int) -> int:
    """Número de qubits para representar inteiros de 0 até n-1."""
    return max(1, math.ceil(math.log2(max(2, n))))


def encode_fixed_point(x: float, frac_bits: int) -> int:
    """Quantiza um número real em fixed-point."""
    return int(round(x * (2 ** frac_bits)))


def apply_match_integer(qc: QuantumCircuit, reg, value: int):
    """
    Prepara controles positivos para reg == value.
    Convenção little-endian.
    """
    bitstr = format(value, f"0{len(reg)}b")[::-1]
    for qb, bit in zip(reg, bitstr):
        if bit == "0":
            qc.x(qb)


def undo_match_integer(qc: QuantumCircuit, reg, value: int):
    """Desfaz apply_match_integer."""
    apply_match_integer(qc, reg, value)


# ============================================================
# 2. DISTÂNCIAS CLÁSSICAS E QUANTIZAÇÃO
# ============================================================

def build_distance_table(S: np.ndarray, frac_bits: int):
    """
    Constrói a tabela Dq[i][j] = d²(s_i, s_j) quantizada.
    """
    S = np.array(S, dtype=float)
    n = len(S)
    Dq = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            d2 = float(np.sum((S[i] - S[j]) ** 2))
            Dq[i][j] = encode_fixed_point(d2, frac_bits)

    return Dq


def max_distance_register_bits(S: np.ndarray, frac_bits: int) -> int:
    """Calcula quantos bits são necessários para armazenar a maior distância quantizada."""
    S = np.array(S, dtype=float)
    n = len(S)

    max_d2 = 0.0
    for i in range(n):
        for j in range(n):
            d2 = float(np.sum((S[i] - S[j]) ** 2))
            max_d2 = max(max_d2, d2)

    max_int = encode_fixed_point(max_d2, frac_bits)
    return max(1, math.ceil(math.log2(max_int + 1)))


# ============================================================
# 3. ORÁCULO DE DISTÂNCIA
# ============================================================

def block_distance_oracle_from_S(qc: QuantumCircuit, p_reg, q_reg, dist_reg, S, frac_bits):
    """
    Implementa:
        |p>|q>|z> -> |p>|q>|z xor Dq[p][q]>
    onde Dq[p][q] é a distância ao quadrado quantizada.
    """
    S = np.array(S, dtype=float)
    n = len(S)
    Dq = build_distance_table(S, frac_bits)
    controls = list(p_reg) + list(q_reg)

    for p in range(n):
        for q in range(n):
            value = Dq[p][q]

            apply_match_integer(qc, p_reg, p)
            apply_match_integer(qc, q_reg, q)

            bitstr = format(value, f"0{len(dist_reg)}b")[::-1]
            for k, bit in enumerate(bitstr):
                if bit == "1":
                    qc.mcx(controls, dist_reg[k])

            undo_match_integer(qc, q_reg, q)
            undo_match_integer(qc, p_reg, p)


# ============================================================
# 4. COMPARATOR COERENTE CONTRA CONSTANTE
# ============================================================

def comparator_leq_constant_oracle(qc: QuantumCircuit, reg_a, threshold_value: int, flag):
    """
    Implementa:
        |a>|0> -> |a>|[a <= threshold_value]>

    Comparator coerente contra constante clássica.
    Custo O(2^b), em vez de O(2^(2b)).
    """
    n_values = 2 ** len(reg_a)

    for a in range(n_values):
        if a <= threshold_value:
            apply_match_integer(qc, reg_a, a)
            qc.mcx(list(reg_a), flag)
            undo_match_integer(qc, reg_a, a)


# ============================================================
# 5. TEMPLATE DO EDGE VALUE (sem fixar p,q)
# ============================================================

def build_edge_value_template_from_S(
    S: np.ndarray,
    d_min: float,
    frac_bits: int = 2,
    uncompute: bool = False,
):
    """
    Constrói o corpo pesado do circuito:
        |p>|q>|0>|0> -> |p>|q>|dist>|a_pq>

    Se uncompute=True:
        |p>|q>|0>|0> -> |p>|q>|0>|a_pq>
    """
    S = np.array(S, dtype=float)
    n = len(S)
    idx_bits = bits_needed(n)
    dist_bits = max_distance_register_bits(S, frac_bits)

    p_reg = QuantumRegister(idx_bits, "p")
    q_reg = QuantumRegister(idx_bits, "q")
    dist_reg = QuantumRegister(dist_bits, "dist")
    adj_reg = QuantumRegister(1, "adj")

    qc = QuantumCircuit(p_reg, q_reg, dist_reg, adj_reg)

    # 1) distance estimation
    block_distance_oracle_from_S(qc, p_reg, q_reg, dist_reg, S, frac_bits)

    # 2) comparator contra constante
    threshold_value = encode_fixed_point(d_min ** 2, frac_bits)
    comparator_leq_constant_oracle(qc, dist_reg, threshold_value, adj_reg[0])

    # 3) opcionalmente limpar dist, preservando adj
    if uncompute:
        block_distance_oracle_from_S(qc, p_reg, q_reg, dist_reg, S, frac_bits)

    return qc


# ============================================================
# 6. PREPARAÇÃO LEVE DOS ÍNDICES p,q
# ============================================================

def build_pair_preparation_circuit(template: QuantumCircuit, p_value: int, q_value: int):
    """
    Cria um circuito leve com Xs para preparar |p>|q>,
    usando os mesmos registradores do template.
    """
    qc = QuantumCircuit(*template.qregs)

    p_reg = next(reg for reg in qc.qregs if reg.name == "p")
    q_reg = next(reg for reg in qc.qregs if reg.name == "q")

    for k, bit in enumerate(format(p_value, f"0{len(p_reg)}b")[::-1]):
        if bit == "1":
            qc.x(p_reg[k])

    for k, bit in enumerate(format(q_value, f"0{len(q_reg)}b")[::-1]):
        if bit == "1":
            qc.x(q_reg[k])

    return qc


def build_edge_value_circuit_from_template(template: QuantumCircuit, p_value: int, q_value: int):
    """
    Compõe:
        preparação leve de p,q + template pesado
    """
    prep = build_pair_preparation_circuit(template, p_value, q_value)
    qc = prep.compose(template)
    return qc


# ============================================================
# 7. EXECUÇÃO COM SHOTS
# ============================================================

def run_edge_value_circuit(qc: QuantumCircuit, shots: int = 256):
    """
    Mede adj e roda diretamente no AerSimulator, sem transpile.
    """
    qc_m = qc.copy()
    c = ClassicalRegister(1, "c")
    qc_m.add_register(c)

    adj_reg = next(reg for reg in qc_m.qregs if reg.name == "adj")
    qc_m.measure(adj_reg[0], c[0])

    backend = AerSimulator()
    result = backend.run(qc_m, shots=shots).result()
    counts = result.get_counts()

    prob1 = counts.get("1", 0) / shots
    return counts, prob1, qc_m


# ============================================================
# 8. MATRIZ A QUÂNTICA REUSANDO O TEMPLATE
# ============================================================

def compute_A_matrix_quantum(S, d_min, frac_bits=2, shots=256, zero_diagonal=False, upper_triangle_only=False):
    """
    Reconstrói A reutilizando um único template pesado.
    """
    S = np.array(S, dtype=float)
    n = len(S)
    A = np.zeros((n, n), dtype=int)

    template = build_edge_value_template_from_S(
        S=S,
        d_min=d_min,
        frac_bits=frac_bits,
        uncompute=False
    )

    for p in range(n):
        q_start = p if upper_triangle_only else 0
        for q in range(q_start, n):
            qc = build_edge_value_circuit_from_template(template, p, q)
            _, prob1, _ = run_edge_value_circuit(qc, shots=shots)
            val = 1 if prob1 > 0.5 else 0

            A[p, q] = val
            if upper_triangle_only:
                A[q, p] = val

    if zero_diagonal:
        np.fill_diagonal(A, 0)

    return A


# ============================================================
# 9. MATRIZ A CLÁSSICA
# ============================================================

def compute_A_matrix_classical(S, d_min, zero_diagonal=False):
    S = np.array(S, dtype=float)
    n = len(S)
    A = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            d2 = np.sum((S[i] - S[j]) ** 2)
            A[i, j] = 1 if d2 <= d_min ** 2 else 0

    if zero_diagonal:
        np.fill_diagonal(A, 0)

    return A


# ============================================================
# 10. DEMO
# ============================================================

if _name_ == "_main_":
    from sklearn.datasets import make_circles
    # ============================================================
    # Geração do dataset
    # ============================================================

    S, y = make_circles(
        n_samples=16,
        noise=0.02,
        factor=0.4,
        random_state=42,
    )
    S = S.astype(float)

    d_min     = 0.75   # menor distância real é d≈0.54 → threshold d²=0.5625
                    # conecta os 8 pares vizinhos mais próximos
    frac_bits = 3      # precisão=0.125, threshold quantizado=5 (folga de 2 LSB)
    shots     = 512

    # constrói o template uma vez
    template = build_edge_value_template_from_S(
        S=S,
        d_min=d_min,
        frac_bits=frac_bits,
        uncompute=False
    )

    print("Número de qubits do template:", template.num_qubits)
    print(template.draw(output="text", fold=120))

    # exemplo para (p,q)=(0,1)
    qc_example = build_edge_value_circuit_from_template(template, 0, 1)
    counts, prob1, _ = run_edge_value_circuit(qc_example, shots=shots)

    print("\nCounts para (p,q)=(0,1):", counts)
    print("Probabilidade de adj=1:", prob1)

    # reconstrução de A
    A_quantum = compute_A_matrix_quantum(
        S=S,
        d_min=d_min,
        frac_bits=frac_bits,
        shots=shots,
        zero_diagonal=False,
        upper_triangle_only=True
    )

    A_classical = compute_A_matrix_classical(
        S=S,
        d_min=d_min,
        zero_diagonal=False
    )

    np.set_printoptions(threshold=np.inf)

    print("\nAdjacency matrix A (quantum):")
    print(A_quantum)

    print("\nAdjacency matrix A (classical):")
    print(A_classical)

    print("\nDiferença A_quantum - A_classical:")
    print(A_quantum - A_classical)