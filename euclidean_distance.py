def euclidean_distance_circuit(a: np.ndarray, b: np.ndarray) -> QuantumCircuit:
    """
  Constrói o circuito para estimativa de d(a, b) conforme
  a construção de referência:

      z    = ‖a‖² + ‖b‖²
      |φ⟩  = 1/√z · (‖a‖|0⟩ - ‖b‖|1⟩)
      |ψ⟩  = 1/√2 · (|0⟩|â⟩ + |1⟩|b̂⟩)

  Estado: |0⟩_anc ⊗ |φ⟩ ⊗ |ψ⟩

  Circuito:
      H(anc)
      CSWAP(anc, φ_qubit, ψ_reg[0])   ← swap entre qubit de φ e 1º de ψ
      H(anc)

  Após medição do ancilla:
      d²(a, b) = 4 · z · (P(anc=0) - 0.5)

  Layout dos qubits (índices globais):
      0        → ancilla do SWAP test
      1        → qubit de |φ⟩  (1 qubit: amplitude ‖a‖ em |0⟩, -‖b‖ em |1⟩)
      2        → qubit de controle de |ψ⟩  (|0⟩→â, |1⟩→b̂)
      3..3+nq-1 → qubits de amplitude de |ψ⟩
  """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = len(a)
    assert len(b) == d, "Vetores devem ter mesma dimensão."

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a > 1e-12 and norm_b > 1e-12, "Vetores zero inválidos."

    normed_a = a / norm_a
    normed_b = b / norm_b
    z = norm_a * 2 + norm_b * 2

    # Número de qubits para amplitude-encoding de dimensão d
    nq = max(1, math.ceil(math.log2(max(2, d))))

    # Amplitudes de |φ⟩ (1 qubit)
    amp_phi = np.array([norm_a, -norm_b]) / math.sqrt(z)

    # Amplitudes de |ψ⟩:  1/√2 · (|0⟩⊗â + |1⟩⊗b̂)
    # Dimensão do espaço: 2 * 2^nq
    amp_a = np.zeros(2 ** nq, dtype=complex)
    amp_b = np.zeros(2 ** nq, dtype=complex)
    amp_a[:d] = normed_a.astype(complex)
    amp_b[:d] = normed_b.astype(complex)

    # |ψ⟩ no espaço (ctrl_qubit ⊗ data_qubits):
    #   |0⟩|â⟩ + |1⟩|b̂⟩  (não normalizado antes do /√2)
    amp_psi = np.concatenate([amp_a, amp_b]) / math.sqrt(2)

    # Registradores
    anc_reg = QuantumRegister(1, "anc_ed")  # ancilla SWAP test
    phi_reg = QuantumRegister(1, "phi")  # estado |φ⟩
    psi_reg = QuantumRegister(1 + nq, "psi")  # ctrl + data de |ψ⟩

    qc = QuantumCircuit(anc_reg, phi_reg, psi_reg)

    # Prepara |φ⟩
    qc.append(
        StatePreparation(amp_phi.astype(complex), label="|phi>"),
        phi_reg,
    )

    # Prepara |ψ⟩ = 1/√2 (|0⟩|â⟩ + |1⟩|b̂⟩)
    qc.append(
        StatePreparation(amp_psi, label="|psi>"),
        psi_reg,
    )

    # SWAP test entre |φ⟩ e o qubit de controle de |ψ⟩
    #   H(anc) → CSWAP(anc, phi[0], psi[0]) → H(anc)
    qc.h(anc_reg[0])
    qc.cswap(anc_reg[0], phi_reg[0], psi_reg[0])
    qc.h(anc_reg[0])

    return qc, z


def estimate_d2_euclidean(
        a: np.ndarray,
        b: np.ndarray,
        shots: int = 2048,
        backend=None,
) -> float:
    """
  Estima d²(a, b) usando o circuito de distância euclidiana.

  Fórmula exata (sem aproximação além do shot noise):
      d²(a, b) = 4 · z · (P(anc=0) - 0.5)

  onde z = ‖a‖² + ‖b‖².
  """
    if backend is None:
        backend = AerSimulator()

    qc, z = euclidean_distance_circuit(a, b)

    # Mede apenas o ancilla (primeiro qubit, índice 0)
    qc_m = qc.copy()
    c = ClassicalRegister(1, "c_ed")
    qc_m.add_register(c)
    qc_m.measure(0, c[0])

    counts = _run(qc_m, shots, backend)
    p0 = counts.get("0", 0) / shots

    d2 = 4.0 * z * (p0 - 0.5)
    return max(0.0, d2)