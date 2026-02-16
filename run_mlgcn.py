#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena MLGCNClassifier y genera predicciones binarias para TEST.

Uso:
  python train_and_predict_gcn.py train.csv test.csv LAST_N_LABELS --seed 42 [--cv]

- LAST_N_LABELS: número de últimas columnas que son etiquetas (en TRAIN y TEST).
- Si --cv: realiza 3-fold CV sobre tau, p, model_kind, proj_dim, mlp_hidden_dims.
- Siempre ajusta umbrales por clase en una validación interna dentro de TRAIN.
- La salida SOLO contiene predicciones binarias (una columna por etiqueta).

Requisitos:
- Módulo: MLGNNClassifiers
- Clase:  MLGCNClassifier
"""

from __future__ import annotations
import argparse
import importlib
import itertools
import json
from typing import List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from contextlib import contextmanager

import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ------------------------------ Utils ------------------------------

def select_label_columns(df: pd.DataFrame, last_n_labels: int) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    if last_n_labels <= 0 or last_n_labels > len(cols):
        raise ValueError("last_n_labels debe estar entre 1 y el número de columnas.")
    y_cols = cols[-last_n_labels:]
    x_cols = cols[:-last_n_labels]
    return x_cols, y_cols


def compute_pos_weight(Y: np.ndarray, cap: float = 100.0) -> np.ndarray:
    Yb = (Y > 0).astype(np.int64)
    N_pos = Yb.sum(axis=0).astype(np.float64)
    N = Yb.shape[0]
    N_neg = N - N_pos
    posw = np.where(N_pos > 0, N_neg / np.maximum(N_pos, 1.0), 1.0)
    posw = np.clip(posw, 1.0, cap)
    return posw.astype(np.float32)


def tune_thresholds_per_class(y_val: np.ndarray, p_val: np.ndarray,
                              grid: Sequence[float] = np.linspace(0.05, 0.95, 19)) -> np.ndarray:
    C = y_val.shape[1]
    thresholds = np.full(C, 0.5, dtype=float)
    for c in range(C):
        y = y_val[:, c]
        p = p_val[:, c]
        if len(np.unique(y)) < 2:
            thresholds[c] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            yhat = (p >= t).astype(int)
            f1 = f1_score(y, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


def make_model(Cls, D_in: int, C: int, params: dict, device: str):

    common_kwargs = {}

    for k in ["input_dim",                
        "device",     
        "proj_dim",            
        "mlp_hidden_dims",        
        "mlp_batchnorm",
        "mlp_dropout",
        "label_embed_dim",
        "gcn_hidden_dims",
        "tau",
        "p",
        "use_class_bias",
        "negative_slope",
        "alpha", ]:
        if k in params:
            common_kwargs[k] = params[k] 
            

    # Instanciación robusta según la firma del constructor
    try:
        return Cls(input_dim=D_in, num_labels=C, **common_kwargs)
    except TypeError as e:
        raise TypeError(f"Error al instanciar {Cls.__name__} con args: {common_kwargs}") from e


def fit_and_eval_once(Cls, X_tr, Y_tr, X_va, Y_va, params: dict, seed: int, device: str) -> float:
    model = make_model(Cls, X_tr.shape[1], Y_tr.shape[1], params, device)
    posw = compute_pos_weight(Y_tr, cap=100.0)
    if not hasattr(model, "fit"):
        raise AttributeError("La clase no tiene método .fit(X, Y, ...)")
    model.fit(X_tr, Y_tr, verbose=False, val=(X_va, Y_va), pos_weight=posw, seed=seed)
    P_va = model.predict_proba(X_va)
    thr = tune_thresholds_per_class(Y_va, P_va)
    Y_hat = (P_va >= thr[None, :]).astype(int)
    f1 = f1_score(Y_va, Y_hat, average="macro", zero_division=0)
    return f1



@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()

# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv", type=str, help="Ruta al CSV de TRAIN.")
    parser.add_argument("test_csv", type=str, help="Ruta al CSV de TEST (mismas columnas).")
    parser.add_argument("last_n_labels", type=int, help="Últimas N columnas son etiquetas.")
    parser.add_argument("output_preds", type=str, help="CSV de salida (solo predicciones).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="Si se activa, hace 3-fold CV de hiperparámetros.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Proporción para validar umbrales.")
    
    parser.add_argument("--n_jobs", type=int, default=-1, help="Número de trabajos paralelos para CV.")
    args = parser.parse_args()

    # Cargar datos

    df_tr = pd.read_csv(args.train_csv, header=None)
    df_te = pd.read_csv(args.test_csv, header=None)

    # X/Y por últimas N columnas
    feat_cols, lab_cols = select_label_columns(df_tr, args.last_n_labels)

    # Comprobar que TEST tiene esas columnas (mismas columnas)
    missing = [c for c in feat_cols + lab_cols if c not in df_te.columns]
    if missing:
        raise ValueError(f"Test no contiene columnas requeridas: {missing}")

    # Matrices
    X_tr_full = df_tr[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    Y_tr_full = (df_tr[lab_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) >= 0.5).astype(np.int64)

    X_te = df_te[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Modelo
    mod = importlib.import_module("MLGNNClassifiers")
    Cls = getattr(mod, "MLGCNClassifier", None)
    if Cls is None:
        raise ImportError("No pude encontrar la clase MLGCNClassifier en MLGNNClassifiers.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Usando dispositivo: {device}")

    # Búsqueda de hiperparámetros si --cv
    if args.cv:
        grid = {
            "tau": [0.2, 0.4, 0.6],
            "p": [0.0, 0.25, 0.5],
            "proj_dim": [64, 128],
            "label_embed_dim": [32, 64, 128],
            "mlp_hidden_dims": [(), (256,), (256, 128)],
            "mlp_dropout": [0.0],
            "gcn_hidden_dims": [(), (128,)],
            "alpha": [1.0],
        }

        keys = list(grid.keys())
        combos = [{k: v for k, v in zip(keys, vals)}
                for vals in itertools.product(*(grid[k] for k in keys))]
        print(f"[CV] Evaluando {len(combos)} combinaciones...")

        kf = KFold(n_splits=3, shuffle=True, random_state=args.seed)
        fold_indices = [(tr, va) for tr, va in kf.split(X_tr_full)]

        def eval_params(params, seed, device_):
            # evitar oversubscription en cada worker
            torch.set_num_threads(1)
            scores = []
            for tr_idx, va_idx in fold_indices:
                X_tr, X_va = X_tr_full[tr_idx], X_tr_full[va_idx]
                Y_tr, Y_va = Y_tr_full[tr_idx], Y_tr_full[va_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_va = scaler.transform(X_va)

                # fit_and_eval_once debe construir el grafo con Y_tr (o llamar a fit que ya lo hace)
                score = fit_and_eval_once(Cls, X_tr, Y_tr, X_va, Y_va, params, seed, device_)
                scores.append(score)
            return float(np.mean(scores)), params

        # n_jobs seguro según dispositivo
        n_jobs_arg = getattr(args, "n_jobs", 1)
        if isinstance(device, torch.device) and device.type == "cuda":
            n_jobs = 1
            print("[CV] CUDA detectado → n_jobs=1 para evitar contención de GPU.")
        else:
            n_jobs = -1 if n_jobs_arg == -1 else n_jobs_arg

        if n_jobs != 1:
            from joblib import Parallel, delayed
            with tqdm_joblib(tqdm(total=len(combos), desc="Grid CV")):
                results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                    delayed(eval_params)(params, args.seed, device) for params in combos
                )
        else:
            results = []
            for params in tqdm(combos, total=len(combos), desc="Grid CV"):
                results.append(eval_params(params, args.seed, device))

        best_score, best_params = max(results, key=lambda t: t[0])
        print(f"[CV] Mejor params: {json.dumps(best_params)} | score={best_score:.4f}")

    else:
        best_params = {
            "tau": 0.4,
            "p": 0.2,
            "model_kind": "linear",
            "proj_dim": 128,
            "mlp_hidden_dims": (256,),
            "label_embed_dim": 32,
            "gcn_hidden_dims": (128,),
      }

    # Ajuste final en TRAIN con split interno SOLO para umbrales
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full, test_size=args.val_size, random_state=args.seed, stratify=None
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # Entrenar
    final_model = make_model(Cls, X_tr.shape[1], Y_tr.shape[1], best_params, device)
    posw = compute_pos_weight(Y_tr, cap=100.0)
    final_model.fit(X_tr, Y_tr, verbose=True, val=(X_va, Y_va), pos_weight=posw, seed=args.seed)

    # Umbrales SIEMPRE validados
    P_va = final_model.predict_proba(X_va)
    thresholds = tune_thresholds_per_class(Y_va, P_va)

    # Predicciones en TEST (solo binarias)
    P_te = final_model.predict_proba(X_te_s)
    Yhat_te = (P_te >= thresholds[None, :]).astype(int)

    # Guardar SOLO predicciones (una columna por etiqueta)
    out = pd.DataFrame({lab_cols[j]: Yhat_te[:, j].astype(int) for j in range(len(lab_cols))})
    out.to_csv(args.output_preds, index=False, header=False)  # <- nombre de salida




if __name__ == "__main__":
    main()
