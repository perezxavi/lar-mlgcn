# pipeline_mlgcn.py
import os
import sys
import re
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd

# ====== 1) Especificaciones de datasets (last_k y si es sparse) ======
@dataclass(frozen=True)
class DatasetSpec:
    last_k: int
    sparse: bool  # True: ARFF {col val,...}; False: ARFF denso (@data CSV)

DATASETS: Dict[str, DatasetSpec] = {
        # Dense
    "cooking":        DatasetSpec(last_k=400, sparse=False),
    "flags":          DatasetSpec(last_k=7,   sparse=False),
    "water-quality-nom":  DatasetSpec(last_k=14,  sparse=False),
    "birds":          DatasetSpec(last_k=19,  sparse=False),
    "CAL500":         DatasetSpec(last_k=174, sparse=False),
    "Corel5k":        DatasetSpec(last_k=374, sparse=False),
    # Sparse
    "Arts1":          DatasetSpec(last_k=26, sparse=True),
    "Business1":      DatasetSpec(last_k=30, sparse=True),
    "Computers1":     DatasetSpec(last_k=33, sparse=True),
    "Education1":     DatasetSpec(last_k=33, sparse=True),
    "Entertainment1": DatasetSpec(last_k=21, sparse=True),
    "Health1":        DatasetSpec(last_k=32, sparse=True),
    "Recreation1":    DatasetSpec(last_k=22, sparse=True),
    "Reference1":     DatasetSpec(last_k=33, sparse=True),
    "Science1":       DatasetSpec(last_k=40, sparse=True),
    "Society1":       DatasetSpec(last_k=27, sparse=True),
    "Social1":        DatasetSpec(last_k=39, sparse=True),

}

# ====== 2) Lectores ARFF ======
# 2.a) Sparse -> DataFrames (features y labels)
import arff                    # liac-arff
from scipy.sparse import csr_matrix

def read_arff_sparse_to_dfs(path: str, last_k_labels: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        ds = arff.load(f)
    attrs = ds["attributes"]
    n_attrs = len(attrs)
    if last_k_labels <= 0 or last_k_labels >= n_attrs:
        raise ValueError(f"last_k_labels={last_k_labels} inválido (n_attrs={n_attrs}).")

    feat_idx = list(range(0, n_attrs - last_k_labels))
    lab_idx  = list(range(n_attrs - last_k_labels, n_attrs))
    feat_map = {orig: j for j, orig in enumerate(feat_idx)}
    lab_map  = {orig: j for j, orig in enumerate(lab_idx)}

    X_data, X_cols, X_indptr = [], [], [0]
    Y_data, Y_cols, Y_indptr = [], [], [0]

    for row in ds["data"]:
        # En ARFF sparse, liac-arff da dict {col_idx: val}
        items = sorted(row.items()) if isinstance(row, dict) else [(i, v) for i, v in enumerate(row) if v not in (0, None, "?")]

        for c, v in items:
            if c in feat_map:
                try:
                    val = float(v)
                except Exception:
                    continue
                if val != 0.0:
                    X_cols.append(feat_map[c]); X_data.append(val)
        X_indptr.append(len(X_cols))

        for c, v in items:
            if c in lab_map:
                try:
                    val = float(v)
                except Exception:
                    val = 1.0
                if val != 0.0:
                    Y_cols.append(lab_map[c]); Y_data.append(1.0)
        Y_indptr.append(len(Y_cols))

    n_rows = len(X_indptr) - 1
    X = csr_matrix((X_data, X_cols, X_indptr), shape=(n_rows, len(feat_idx)))
    Y = csr_matrix((Y_data, Y_cols, Y_indptr), shape=(n_rows, len(lab_idx)))

    X_df = pd.DataFrame.sparse.from_spmatrix(X)
    Y_df = pd.DataFrame.sparse.from_spmatrix(Y).astype(pd.SparseDtype("int", fill_value=0))

    return X_df, Y_df

# 2.b) Denso -> DataFrames (features y labels)
from io import StringIO
def read_arff_dense_to_dfs(path: str, last_k_labels: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    attr_names, data_lines, data_started = [], [], False
    attr_re = re.compile(r"""(?i)^\s*@attribute\s+(?:'([^']+)'|"([^"]+)"|([^\s]+))\s+(.+?)\s*$""", re.VERBOSE)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"): 
                continue
            if not data_started:
                m = attr_re.match(line)
                if m:
                    name = m.group(1) or m.group(2) or m.group(3)
                    attr_names.append(name); continue
                if line.lower().startswith("@data"):
                    data_started = True; continue
            else:
                if line: data_lines.append(line)

    if not attr_names: raise ValueError("No se encontraron @attribute en el header.")
    if not data_lines: raise ValueError("No se encontraron datos tras @data.")
    if last_k_labels <= 0 or last_k_labels >= len(attr_names):
        raise ValueError(f"last_k_labels={last_k_labels} inválido (n_cols={len(attr_names)}).")

    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=attr_names,
                     comment="%", na_values=["?"], engine="python")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X_df = df.iloc[:, :-last_k_labels].copy()
    Y_df = df.iloc[:, -last_k_labels:].copy()
    # normaliza Y a {0,1} ints
    for c in Y_df.columns:
        Y_df[c] = pd.to_numeric(Y_df[c], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    return X_df, Y_df

# ====== 3) Carga unificada a DataFrames ======
def load_dataset_to_dfs(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], int]:
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' no está en DATASETS.")
    spec = DATASETS[name]
    path = os.path.join("datasets", f"{name}.arff")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")
    if spec.sparse:
        X_df, Y_df = read_arff_sparse_to_dfs(path, spec.last_k)
    else:
        X_df, Y_df = read_arff_dense_to_dfs(path, spec.last_k)
    return X_df, Y_df, spec.last_k

# ====== 4) Hold-out aleatorio y guardado a CSVs temporales ======
from sklearn.model_selection import train_test_split

def make_holdout_and_csvs(X_df: pd.DataFrame, Y_df: pd.DataFrame, last_k: int,
    tmp_dir: str, test_size: float = 0.2, random_state: int = 42 ) -> Tuple[str, str, str, List[str]]:
    """
    Crea hold-out aleatorio y guarda:
      - train_tmp.csv : features + labels (sin cabecera)
      - test_tmp.csv  : features + labels (sin cabecera)
      - test.csv      : SOLO etiquetas de test (sin cabecera)
    Devuelve: (train_csv, test_csv, gt_csv)
    """
    # Combina en orden: X luego Y
    full_df = pd.concat(
        [X_df.reset_index(drop=True), Y_df.reset_index(drop=True)],
        axis=1
    )

    n_cols = full_df.shape[1]
    if last_k <= 0 or last_k >= n_cols:
        raise ValueError(f"last_k={last_k} inválido para n_cols={n_cols}")

    # Reindexa columnas por posición para evitar nombres
    full_df.columns = range(n_cols)

    # Hold-out
    train_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Rutas
    os.makedirs(tmp_dir, exist_ok=True)
    train_csv = os.path.join(tmp_dir, "train_tmp.csv")
    test_csv  = os.path.join(tmp_dir, "test_tmp.csv")
    gt_csv    = os.path.join(tmp_dir, "test.csv")  # etiquetas de test

    # Guarda sin cabecera
    train_df.to_csv(train_csv, index=False, header=False)
    test_df.to_csv(test_csv,   index=False, header=False)

    # Últimas last_k columnas son las etiquetas
    test_df.iloc[:, -last_k:].to_csv(gt_csv, index=False, header=False)

    return train_csv, test_csv, gt_csv

# ====== 5) Llamada a run_mlgcn.py ======
def call_run_mlgcn( train_csv: str, test_csv: str, last_k: int, pred_csv: str, cv: bool = False, n_jobs: int = -1):
    # usa el mismo intérprete de Python que ejecuta este script
    cmd = [sys.executable, "run_mlgcn.py", train_csv, test_csv, str(last_k), "--output-preds" , pred_csv, "--n_jobs", str(n_jobs)]
    if cv:
        cmd.append("--cv")

    print(">> Ejecutando:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# ====== 6) Evaluación de métricas ======
import numpy as np
from sklearn.metrics import (
    hamming_loss, accuracy_score,
    precision_score, recall_score, f1_score
)


def evaluate_predictions(pred_csv: str, gt_csv: str) -> dict:
    y_true = pd.read_csv(gt_csv)
    y_pred = pd.read_csv(pred_csv)

    Yt = y_true.values
    Yp = y_pred.values

    metrics = {
        "subset_accuracy": float(accuracy_score(Yt, Yp)),
        "hamming_loss":   float(hamming_loss(Yt, Yp)),
        "precision_micro": float(precision_score(Yt, Yp, average="micro", zero_division=0)),
        "recall_micro":    float(recall_score(Yt, Yp, average="micro", zero_division=0)),
        "f1_micro":        float(f1_score(Yt, Yp, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(Yt, Yp, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(Yt, Yp, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(Yt, Yp, average="macro", zero_division=0)),
    }
    return metrics

# ====== 7) Orquestador principal ======
def run_pipeline(dataset_name: str,
                 test_size: float = 0.2, 
                 random_state: int = 42, 
                 just_performance: bool = False, 
                 cv: bool = False,
                 n_jobs: int = -1) -> dict:
    
    if not just_performance:
        X_df, Y_df, last_k = load_dataset_to_dfs(dataset_name)

        # carpeta temporal
        os.makedirs(f'tmp_mlgcn_{dataset_name}', exist_ok=True)
        pred_csv = os.path.join(f'tmp_mlgcn_{dataset_name}', "predictions.csv")

        # (1) Hold-out + (2) CSVs
        train_csv, test_csv, gt_csv = make_holdout_and_csvs(X_df, Y_df, last_k, f'tmp_mlgcn_{dataset_name}',
                                                                        test_size=test_size,
                                                                        random_state=random_state)

        # (3) Llamar a run_mlgcn.py
        call_run_mlgcn(train_csv, test_csv, last_k, pred_csv, cv=cv, n_jobs=n_jobs)

    # (4) Evaluación
    tmp_dir = os.path.join(os.getcwd(), f"tmp_mlgcn_{dataset_name}")
    pred_csv = os.path.join(tmp_dir, "predictions.csv")
    gt_csv   = os.path.join(tmp_dir, "test.csv")


    metrics = evaluate_predictions(pred_csv, gt_csv)

    print("\n== Resultados ==")
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.6f}")
    print("==============\n\n")
    """
    return {
        "train_csv": train_csv,
        "test_csv": test_csv,
        "gt_csv": gt_csv,
        "pred_csv": pred_csv,
        "metrics": metrics,
    }
    """
 
# ====== 8) CLI simple ======
if __name__ == "__main__":
    cv = True
    n_jobs = 5
    for ds in DATASETS:
        print(f" - {ds}: last_k={DATASETS[ds].last_k}, sparse={DATASETS[ds].sparse}")
        tmp_dir = os.path.join(os.getcwd(), f"tmp_mlgcn_{ds}")
        pred_csv = os.path.join(tmp_dir, "predictions.csv")
        gt_csv   = os.path.join(tmp_dir, "test.csv")
        if os.path.exists(pred_csv) and os.path.exists(gt_csv):
            print("   Ya existen predicciones y ground truth, se omite.")
            run_pipeline(ds, just_performance=True)

        else:
            run_pipeline(ds, just_performance=False, cv=cv, n_jobs=n_jobs)

