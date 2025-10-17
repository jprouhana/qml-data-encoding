"""
VQC classification with different data encoding strategies.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import COBYLA
from .encoders import angle_encoding, dense_angle_encoding


def build_vqc_with_encoding(encoding_type, n_features, n_layers=2):
    """
    Build a VQC with specified encoding followed by trainable ansatz.

    Args:
        encoding_type: 'angle' or 'dense_angle'
        n_features: number of input features
        n_layers: variational layers

    Returns:
        circuit, data_params, trainable_params
    """
    if encoding_type == 'angle':
        enc_circuit, data_params = angle_encoding(n_features)
        n_qubits = n_features
    elif encoding_type == 'dense_angle':
        enc_circuit, data_params = dense_angle_encoding(n_features)
        n_qubits = enc_circuit.num_qubits
    else:
        raise ValueError(f"Unknown encoding: {encoding_type}")

    # trainable ansatz
    n_train_params = n_qubits * 2 * n_layers
    train_params = ParameterVector('w', n_train_params)

    ansatz = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            ansatz.ry(train_params[idx], q)
            idx += 1
            ansatz.rz(train_params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            ansatz.cx(q, q + 1)

    full_circuit = enc_circuit.compose(ansatz)
    full_circuit.measure_all()

    return full_circuit, data_params, train_params


def classify_sample(circuit, data_params, train_params, x, weights,
                     shots=1024, seed=42):
    """Run a single classification and return predicted label."""
    param_dict = dict(zip(data_params, x))
    param_dict.update(dict(zip(train_params, weights)))
    bound = circuit.assign_parameters(param_dict)

    backend = AerSimulator(seed_simulator=seed)
    job = backend.run(bound, shots=shots)
    counts = job.result().get_counts()

    # label from first qubit majority vote
    count_0 = sum(v for k, v in counts.items() if k[-1] == '0')
    count_1 = shots - count_0
    return 0 if count_0 > count_1 else 1


def train_vqc(circuit, data_params, train_params, X_train, y_train,
               maxiter=60, shots=512, seed=42):
    """
    Train a VQC using COBYLA on the training data.

    Returns dict with 'weights', 'cost_history'
    """
    cost_history = []

    def cost_function(weights):
        correct = 0
        for x, y in zip(X_train, y_train):
            pred = classify_sample(circuit, data_params, train_params,
                                    x, weights, shots=shots, seed=seed)
            if pred == y:
                correct += 1
        accuracy = correct / len(y_train)
        cost = 1 - accuracy
        cost_history.append(cost)
        return cost

    n_weights = len(train_params)
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=n_weights)

    optimizer = COBYLA(maxiter=maxiter)
    result = optimizer.minimize(cost_function, x0=x0)

    return {
        'weights': result.x,
        'cost_history': cost_history,
    }


def evaluate_vqc(circuit, data_params, train_params, weights,
                  X_test, y_test, shots=1024, seed=42):
    """Evaluate trained VQC accuracy on test set."""
    correct = 0
    for x, y in zip(X_test, y_test):
        pred = classify_sample(circuit, data_params, train_params,
                                x, weights, shots=shots, seed=seed)
        if pred == y:
            correct += 1
    return correct / len(y_test)
