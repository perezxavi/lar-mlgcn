#!/usr/bin/env python3
"""
evaluate_gcn_performance.py

Evalúa el rendimiento de un clasificador multi-etiqueta a partir de:
  - CSV de TEST con las etiquetas verdaderas (Y)
  - CSV de PREDICCIONES con columnas *_proba (y opcionales *_pred)

Métricas:
  - Globales: mAP (mean AP), AUROC micro/macro, F1/Precision/Recall micro/macro/weighted
  - Por clase: AP, AUROC, F1, Precision, Recall, Soporte, Prevalencia, TPRatePred, TP/FP/TN/FN

Uso:
  python evaluate_gcn_performance.py test.csv preds_test.csv \
      --label-prefix y_ \
      --thresholds-json thresholds.json \
      --id-col id \
      --out-json metrics_summary.json \
      --out-perclass-csv metrics_per_class.csv

Requisitos: pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import argparse
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def select_label_columns(df: pd.DataFrame,
                         last_n_labels: Optional[int]) -> List[str]:
    cols = list(df.columns)
    labels = cols[-last_n_labels:]
    return labels


def per_class_confusion(y_true_c: np.ndarray, y_pred_c: np.ndarray) -> Tuple[int, int, int, int]:
    # TP, FP, TN, FN
    tp = int(((y_true_c == 1) & (y_pred_c == 1)).sum())
    fp = int(((y_true_c == 0) & (y_pred_c == 1)).sum())
    tn = int(((y_true_c == 0) & (y_pred_c == 0)).sum())
    fn = int(((y_true_c == 1) & (y_pred_c == 0)).sum())
    return tp, fp, tn, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_csv", type=str, help="CSV con Y verdaderas (test).")
    parser.add_argument("preds_csv", type=str, help="CSV con columnas *_proba y opcionales *_pred.")
    parser.add_argument("last_n_labels", type=int, default=None, help="Últimas N columnas como Y.")
    parser.add_argument("--delimiter", type=str, default=",")
   
    parser.add_argument("--thresholds-json", type=str, default=None,
                        help="JSON con 'thresholds' por clase (si no hay *_pred).")
    parser.add_argument("--out-json", type=str, default="metrics_summary.json")
    parser.add_argument("--out-perclass-csv", type=str, default="metrics_per_class.csv")
    args = parser.parse_args()

    # 1) Cargar datos
    df_te = pd.read_csv(args.test_csv, sep=args.delimiter)
    df_pr = pd.read_csv(args.preds_csv, sep=args.delimiter)

    # 2) Determinar columnas de etiquetas
    lab_cols = select_label_columns(df_te, args.last_n_labels)
    C = len(lab_cols)


        # Alinear por posición
    if len(df_te) != len(df_pr):
        raise ValueError("Sin id-col, test y preds deben tener el MISMO número de filas.")
    df = pd.concat([df_te[lab_cols].reset_index(drop=True),
                   df_pr.reset_index(drop=True)], axis=1)

    # 4) Matrices verdad y probabilidades
    Y_true = (df[lab_cols].to_numpy(dtype=float) > 0.5).astype(np.int64)

    proba_cols = [f"{c}_proba" for c in lab_cols]
    if not all(c in df.columns for c in proba_cols):
        raise ValueError("El CSV de predicciones no contiene columnas *_proba para todas las etiquetas.")
    P = df[proba_cols].to_numpy(dtype=float)

    # 5) Binarización: usar *_pred si existen; si no, thresholds.json; si no, 0.5
    pred_cols = [f"{c}_pred" for c in lab_cols]
    if all(c in df.columns for c in pred_cols):
        Y_pred = df[pred_cols].to_numpy(dtype=int)
        thresholds_used = None
    else:
        if args.thresholds_json:
            with open(args.thresholds_json, "r", encoding="utf-8") as f:
                tj = json.load(f)
            thr = np.array(tj.get("thresholds", []), dtype=float)
            if thr.shape[0] != C:
                print("[AVISO] thresholds.json no coincide en nº de clases; uso umbral 0.5.")
                thr = np.full(C, 0.5, dtype=float)
        else:
            thr = np.full(C, 0.5, dtype=float)
        Y_pred = (P >= thr[None, :]).astype(int)
        thresholds_used = thr

    # 6) Métricas globales
    results = {}
    # mAP (mean AP por clase)
    try:
        ap_per_class = average_precision_score(Y_true, P, average=None)
        results["map"] = float(np.nanmean(ap_per_class))
    except Exception:
        results["map"] = float("nan")

    # AUROC micro/macro
    try:
        results["auroc_micro"] = roc_auc_score(Y_true.reshape(-1), P.reshape(-1))
    except Exception:
        results["auroc_micro"] = float("nan")
    try:
        results["auroc_macro"] = roc_auc_score(Y_true, P, average="macro")
    except Exception:
        results["auroc_macro"] = float("nan")

    # F1/Prec/Rec (micro/macro/weighted) sobre Y_pred
    for avg in ["micro", "macro", "weighted"]:
        results[f"f1_{avg}"] = f1_score(Y_true, Y_pred, average=avg, zero_division=0)
        results[f"precision_{avg}"] = precision_score(Y_true, Y_pred, average=avg, zero_division=0)
        results[f"recall_{avg}"] = recall_score(Y_true, Y_pred, average=avg, zero_division=0)

    # 7) Métricas por clase
    rows = []
    for j, lab in enumerate(lab_cols):
        yt = Y_true[:, j]
        pp = P[:, j]
        yp = Y_pred[:, j]

        support = int(yt.sum())
        prevalence = float(yt.mean())
        pred_rate = float(yp.mean())

        # AP
        try:
            ap = float(average_precision_score(yt, pp))
        except Exception:
            ap = float("nan")
        # AUROC
        try:
            au = float(roc_auc_score(yt, pp))
        except Exception:
            au = float("nan")

        f1 = f1_score(yt, yp, zero_division=0)
        pr = precision_score(yt, yp, zero_division=0)
        rc = recall_score(yt, yp, zero_division=0)
        tp, fp, tn, fn = per_class_confusion(yt, yp)

        rows.append({
            "label": lab,
            "support_pos": support,
            "prevalence": prevalence,
            "pred_positive_rate": pred_rate,
            "AP": ap,
            "AUROC": au,
            "F1": float(f1),
            "Precision": float(pr),
            "Recall": float(rc),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })

    per_class_df = pd.DataFrame(rows)

    # 8) Mostrar resumen
    print("\n=== MÉTRICAS GLOBALES (TEST) ===")
    for k in ["map","auroc_micro","auroc_macro","f1_micro","f1_macro","precision_micro","recall_micro"]:
        v = results.get(k, None)
        if v is not None:
            print(f"{k:>14s}: {v:.4f}" if isinstance(v, float) else f"{k:>14s}: {v}")

    if thresholds_used is not None:
        print("\n(Se usaron umbrales por clase de JSON o 0.5 para binarizar; no había *_pred.)")

    # 9) Guardar a disco
    summary = {
        "labels": lab_cols,
        "metrics_global": results,
        "notes": "Si thresholds_used es null, se usaron *_pred; de lo contrario, se binarizó con thresholds.json o 0.5.",
        "thresholds_used": None if thresholds_used is None else thresholds_used.tolist(),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    per_class_df.to_csv(args.out_perclass_csv, index=False)

    print(f"\nGuardado resumen global en: {args.out_json}")
    print(f"Guardado métricas por clase en: {args.out_perclass_csv}")


if __name__ == "__main__":
    main()
