"""
Quantum data encoding circuit implementations.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


def basis_encoding(data):
    """
    Basis encoding: encode binary features as computational basis states.
    Each feature maps to one qubit: 0 -> |0>, 1 -> |1>.
    Requires binary input.
    """
    n_features = len(data)
    qc = QuantumCircuit(n_features)
    for i, val in enumerate(data):
        if val > 0.5:
            qc.x(i)
    return qc


def angle_encoding(n_features):
    """
    Angle encoding: encode each feature as a rotation angle.
    One qubit per feature, single layer of Ry rotations.
    """
    params = ParameterVector('x', n_features)
    qc = QuantumCircuit(n_features)
    for i in range(n_features):
        qc.ry(params[i], i)
    return qc, params


def dense_angle_encoding(n_features):
    """
    Dense angle encoding: encode two features per qubit using
    Ry and Rz rotations. Halves the qubit requirement.
    """
    n_qubits = int(np.ceil(n_features / 2))
    params = ParameterVector('x', n_features)
    qc = QuantumCircuit(n_qubits)

    idx = 0
    for q in range(n_qubits):
        if idx < n_features:
            qc.ry(params[idx], q)
            idx += 1
        if idx < n_features:
            qc.rz(params[idx], q)
            idx += 1

    return qc, params


def amplitude_encoding(n_features):
    """
    Amplitude encoding: encode the data vector as amplitudes of
    the quantum state. Requires log2(n_features) qubits but
    deeper circuits for state preparation.

    Uses Qiskit's initialize method for exact preparation.
    """
    n_qubits = int(np.ceil(np.log2(n_features)))
    dim = 2 ** n_qubits

    qc = QuantumCircuit(n_qubits)

    # placeholder — actual data is bound at runtime
    # using initialize for exact amplitude preparation
    return qc, n_qubits, dim


def amplitude_encode_data(data):
    """
    Create a circuit that encodes a specific data vector into amplitudes.
    Normalizes the data vector to unit length.
    """
    n_qubits = int(np.ceil(np.log2(len(data))))
    dim = 2 ** n_qubits

    # pad and normalize
    padded = np.zeros(dim)
    padded[:len(data)] = data
    norm = np.linalg.norm(padded)
    if norm > 0:
        padded = padded / norm

    qc = QuantumCircuit(n_qubits)
    qc.initialize(padded, range(n_qubits))
    return qc


def get_encoders():
    """Return dict of encoder names and builder functions."""
    return {
        'Basis': basis_encoding,
        'Angle': angle_encoding,
        'Dense Angle': dense_angle_encoding,
        'Amplitude': amplitude_encode_data,
    }
