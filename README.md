# Quantum Spectral Clustering Pipeline

This project implements a **hybrid/quantum spectral clustering pipeline**, combining quantum distance estimation, graph construction, incidence matrix generation, Laplacian computation, and spectral projection using **VQD**.

## File structure

### `euclidean_distance.py`
Computes the squared-distance matrix between data points.

- Implements the **quantum** version using the **Hadamard test**
- Also includes the **classical** version for comparison

### `edges.py`
Builds the adjacency matrix from the distance matrix.

- Uses a **quantum comparator** to check whether \( d^2 \le d_{\max}^2 \)
- Also includes the classical adjacency construction

### `incidence_matrix.py`
Builds the incidence-like matrix (or matrix `B`) from the graph adjacency matrix.

- Serves as an intermediate step for Laplacian construction
- Contains both the classical and the quantum-inspired/quantum version of `B`

### `laplacian_matrix.py`
Computes the Laplacian-like matrix in the form

\[
L = B B^T
\]

- Implements the **classical** version
- Implements the **quantum** version by estimating inner products with the Hadamard test

### `spectral_space.py`
Solves the spectral problem for the Laplacian.

- Pads the matrix to a power-of-two dimension
- Converts the matrix into a quantum operator
- Uses **VQD (Variational Quantum Deflation)** to estimate the smallest eigenvalues/eigenvectors
- Builds the spectral embedding and applies **KMeans**

### `spectral_clustering.py`
Main file for the complete pipeline.

It executes the following steps:

1. Dataset generation
2. Distance matrix computation
3. Adjacency matrix construction
4. Incidence matrix `B`
5. Laplacian `L`
6. Spectral space computation
7. Clustering with KMeans
8. Final metrics

### `graph_test.py`
Auxiliary file for testing and visualizing the classical and quantum graphs.

- Generates synthetic datasets
- Compares adjacency matrices
- Shows edge differences
- Plots both graphs side by side

## Dependencies

Install the required libraries with:

```bash
pip install numpy matplotlib scikit-learn qiskit qiskit-aer qiskit-algorithms
