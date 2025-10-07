"""
Circuit resource analysis for different encoding strategies.
"""

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from .encoders import angle_encoding, dense_angle_encoding, amplitude_encode_data


def count_resources(circuit):
    """
    Count gates and circuit depth after transpilation.

    Returns dict with 'depth', 'gate_count', 'cx_count', 'n_qubits'
    """
    backend = AerSimulator()
    transpiled = transpile(circuit, backend, optimization_level=2)

    gate_count = sum(transpiled.count_ops().values())
    cx_count = transpiled.count_ops().get('cx', 0)

    return {
        'depth': transpiled.depth(),
        'gate_count': gate_count,
        'cx_count': cx_count,
        'n_qubits': transpiled.num_qubits,
    }


def compare_encoding_resources(n_features_range, seed=42):
    """
    Compare circuit resources for different encodings across feature counts.

    Returns:
        dict mapping encoder name to list of resource dicts
    """
    rng = np.random.RandomState(seed)
    results = {'Angle': [], 'Dense Angle': [], 'Amplitude': []}

    for n_features in n_features_range:
        data = rng.uniform(0, np.pi, size=n_features)

        # angle encoding
        qc_angle, params = angle_encoding(n_features)
        bound = qc_angle.assign_parameters(dict(zip(params, data)))
        results['Angle'].append(count_resources(bound))

        # dense angle encoding
        qc_dense, params_d = dense_angle_encoding(n_features)
        bound_d = qc_dense.assign_parameters(dict(zip(params_d, data)))
        results['Dense Angle'].append(count_resources(bound_d))

        # amplitude encoding
        qc_amp = amplitude_encode_data(data)
        results['Amplitude'].append(count_resources(qc_amp))

    return results


def resource_summary_table(n_features_range, resources):
    """Print a formatted resource comparison table."""
    print(f"{'Encoding':<15} {'Features':<10} {'Qubits':<8} {'Depth':<8} "
          f"{'Gates':<8} {'CX':<6}")
    print('-' * 55)

    for name in resources:
        for i, n_f in enumerate(n_features_range):
            r = resources[name][i]
            print(f"{name:<15} {n_f:<10} {r['n_qubits']:<8} {r['depth']:<8} "
                  f"{r['gate_count']:<8} {r['cx_count']:<6}")
