# Latent and Robust Multi-label Graph Convolutional Network (LaR-MLGCN)

PyTorch implementation of **LaR-MLGCN**, an approach for multi-label classification in tabular data that explicitly models label dependencies through a graph structure.

## Abstract

Multi-label classification poses the critical challenge of modeling complex dependencies between categories in high-dimensional output spaces. Despite recent advances, traditional approaches often fail to capture these correlations explicitly or lack the efficiency required to scale.

In this work, we propose an architecture based on Graph Convolutional Networks (GCN) that shifts the core of learning toward a relational label space. The approach employs a GCN to dynamically generate parameterized classifiers, integrating semantic information derived from the data's co-occurrence topology, reinforced by a density-sensitive adaptive regularization policy.

Comprehensive evaluation across 19 benchmark datasets demonstrates that the proposed model consistently outperforms the state-of-the-art, achieving top average ranks across different performance metrics. The results, statistically validated, confirm that the integration of an explicit relational structure not only maximizes predictive capacity but also offers a structurally more efficient and interpretable alternative to classifier ensemble methods.

## Key Contributions

- **Explicit modeling** of label dependencies via co-occurrence graphs.
- **Classifier generation** per label through GCN propagation.
- **Adaptive regularization policy** based on graph density/degree.
- **Efficient and interpretable design** for tabular multi-label classification.

## Architecture

The architecture combines two main paths:

1. **Label Path (GCN)**
   - Constructs a label graph from `Y_train`.
   - Propagates latent label embeddings to produce a classifier matrix `W`.

2. **Data Path (MLP)**
   - Projects input features `X` into a latent space `Phi`.

3. **Prediction**
   - Calculates logits as `Phi @ W.T` (optionally including class-wise bias).

## Installation

Requirements:

- Python 3.10+
- Dependencies defined in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from MLGNNClassifiers import LaRMLGCNClassifier

X_train = np.random.randn(200, 20).astype(np.float32)
Y_train = np.random.randint(0, 2, size=(200, 5)).astype(np.int64)
X_test = np.random.randn(50, 20).astype(np.float32)

model = LaRMLGCNClassifier(
    input_dim=20,
    num_labels=5,
    proj_dim=128,
    mlp_hidden_dims=(256,),
    label_embed_dim=32,
    gcn_hidden_dims=(128,),
    tau=0.4,
    p=0.2,
)

model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=True)
proba = model.predict_proba(X_test)              # (50, 5), range [0, 1]
pred = model.predict(X_test, threshold=0.5)      # (50, 5), binary output
```

## Training and Evaluation

Main training/prediction script:

```bash
python run_mlgcn.py <train_csv> <test_csv> <last_n_labels> <output_preds> --seed 42 --cv
```

Where:

- `train_csv`: Training dataset.
- `test_csv`: Test dataset.
- `last_n_labels`: Number of final columns corresponding to labels.
- `output_preds`: Output file for binary predictions.
- `--cv`: Enables cross-validation search.

## Reproducibility

Unit tests and lightweight integration:

```bash
python -m unittest test_mlgcn_classifier.py
```

On Windows with Anaconda:

```powershell
python.exe -m unittest test_mlgcn_classifier.py
```

## Repository Structure

- `MLGNNClassifiers.py`: Implementation of `MLGCNClassifier`.
- `run_mlgcn.py`: Pipeline for training, threshold tuning, and prediction.
- `evaluate_mlgcn_performance.py`: Evaluation utilities.
- `test_mlgcn_classifier.py`: Unit and integration tests.
- `requirements.txt`: Dependencies.

## Citation

If this repository is useful for your research, please cite the associated work.
You can replace the following entry when final metadata is available:

```bibtex
@article{larmlgcn2026,
  title   = {Latent and Robust Multi-label Graph Convolutional Network},
  author  = {Perez, ...},
  journal = {Under review},
  year    = {2026}
}
```

## License

Define your project license here (e.g., `MIT`, `Apache-2.0`, or `GPL-3.0`) and include the corresponding `LICENSE` file.
