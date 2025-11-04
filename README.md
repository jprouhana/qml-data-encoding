# Quantum Data Encoding Strategies

Systematic comparison of data encoding methods for quantum machine learning, measuring their impact on circuit resources and downstream classification performance.

## Overview

Data encoding is a critical bottleneck in QML — the choice of encoding determines what the quantum circuit can learn. This project implements and compares four encoding strategies: basis encoding, amplitude encoding, angle encoding, and dense angle encoding, evaluating each on circuit depth, gate count, and classification accuracy.

## Structure

```
src/
  encoders.py          # Basis, amplitude, angle, dense angle encoders
  resource_analysis.py # Gate count, circuit depth, qubit requirements
  classification.py    # VQC training with each encoding strategy
  data_utils.py        # Dataset loading and preprocessing
  plotting.py          # Resource comparison and accuracy plots
notebooks/
  encoding_comparison.ipynb  # Full analysis notebook
```

## Key Results

| Encoding    | Qubits (4 features) | Depth | Accuracy (Iris) |
|------------|---------------------|-------|-----------------|
| Basis      | 4                   | 4     | 0.72            |
| Angle      | 4                   | 1     | 0.89            |
| Dense Angle| 2                   | 1     | 0.91            |
| Amplitude  | 2                   | 8     | 0.87            |

## References

- LaRose & Coyle, "Robust data encodings for quantum classifiers" (2020)
- Schuld & Petruccione, "Supervised Learning with Quantum Computers" (2018)

## Requirements

```
pip install -r requirements.txt
```
