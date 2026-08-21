"""Combine completed case results and compute one-factor intervention contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def select(df, **kwargs):
    q = df.copy()
    for key, value in kwargs.items():
        q = q[q[key] == value]
    return q.iloc[0] if len(q) else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="summary"); args = p.parse_args()
    root, out = Path(args.results_dir), Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in root.glob("*/*/configuration_results.csv"):
        f = pd.read_csv(path); f.insert(0, "dataset", path.parent.name); f.insert(1, "transform", path.parent.parent.name); frames.append(f)
    if not frames:
        raise SystemExit(f"No completed configuration results beneath {root}")
    all_results = pd.concat(frames, ignore_index=True); all_results.to_csv(out / "all_configuration_results.csv", index=False)

    lambda_frames = []
    for path in root.glob("*/*/lambda_path_diagnostics.csv"):
        f = pd.read_csv(path); f.insert(0, "dataset", path.parent.name); f.insert(1, "transform", path.parent.parent.name); lambda_frames.append(f)
    if lambda_frames:
        all_lambda = pd.concat(lambda_frames, ignore_index=True)
        all_lambda.to_csv(out / "all_lambda_path_diagnostics.csv", index=False)
        numerical_columns = [
            "transform", "dataset", "centered", "dtype", "n_train",
            "n_features_raw", "n_features_used", "n_low_variance_dropped",
            "feature_mean_abs_average", "feature_mean_max_abs", "solve_space",
            "gram_min_eigenvalue_raw", "gram_max_eigenvalue_raw",
            "gram_nonpositive_eigenvalues", "gram_condition_positive",
            "decomposition_seconds",
        ]
        (all_lambda[numerical_columns].drop_duplicates()
         .sort_values(["transform", "dataset", "centered", "dtype"])
         .to_csv(out / "numerical_stability_summary.csv", index=False))
    contrasts = []
    for (transform, dataset), g in all_results.groupby(["transform", "dataset"]):
        base_keys = dict(model="DiagnosticPreVal", centered=False, dtype="float32",
                         calibration_constraint="unconstrained", lambda_source="preval",
                         decision_rule="clipped_softmax")
        current = select(g, **base_keys)
        interventions = {
            "raw_score_argmax": {**base_keys, "decision_rule": "raw_score"},
            "centering": {**base_keys, "centered": True},
            "float64": {**base_keys, "dtype": "float64"},
            "positive_scale": {**base_keys, "calibration_constraint": "positive"},
            "ridge_selected_lambda": {**base_keys, "lambda_source": "ridge"},
            "fixed_lambda_1": {**base_keys, "lambda_source": "fixed"},
            "combined_repair_raw": dict(model="DiagnosticPreVal", centered=True, dtype="float64",
                                        calibration_constraint="positive", lambda_source="preval", decision_rule="raw_score"),
            "combined_repair_probability": dict(model="DiagnosticPreVal", centered=True, dtype="float64",
                                                calibration_constraint="positive", lambda_source="preval", decision_rule="clipped_softmax"),
        }
        for name, keys in interventions.items():
            row = select(g, **keys)
            if current is not None and row is not None:
                contrasts.append({"transform": transform, "dataset": dataset, "intervention": name,
                                  "current_accuracy": current.accuracy, "intervention_accuracy": row.accuracy,
                                  "accuracy_change_from_current": row.accuracy-current.accuracy,
                                  "current_tied_row_fraction": current.fraction_rows_tied_max,
                                  "intervention_tied_row_fraction": row.fraction_rows_tied_max,
                                  "current_scale": current.scale, "intervention_scale": row.scale,
                                  "current_lambda": current.selected_lambda, "intervention_lambda": row.selected_lambda})
    pd.DataFrame(contrasts).to_csv(out / "intervention_contrasts.csv", index=False)

    # Compact rows used for the main mechanism interpretation.
    headline = []
    recipes = {
        "current_preval": dict(model="DiagnosticPreVal", centered=False, dtype="float32",
                                calibration_constraint="unconstrained", lambda_source="preval",
                                decision_rule="clipped_softmax"),
        "current_fit_raw_argmax": dict(model="DiagnosticPreVal", centered=False, dtype="float32",
                                        calibration_constraint="unconstrained", lambda_source="preval",
                                        decision_rule="raw_score"),
        "centered_float32": dict(model="DiagnosticPreVal", centered=True, dtype="float32",
                                  calibration_constraint="unconstrained", lambda_source="preval",
                                  decision_rule="clipped_softmax"),
        "uncentered_float64": dict(model="DiagnosticPreVal", centered=False, dtype="float64",
                                    calibration_constraint="unconstrained", lambda_source="preval",
                                    decision_rule="clipped_softmax"),
        "centered_float64": dict(model="DiagnosticPreVal", centered=True, dtype="float64",
                                  calibration_constraint="unconstrained", lambda_source="preval",
                                  decision_rule="clipped_softmax"),
        "ridgecv_uncentered": dict(model="RidgeClassifierCV", centered=False),
        "ridgecv_centered": dict(model="RidgeClassifierCV", centered=True),
    }
    keep = ["accuracy", "selected_lambda", "scale", "nll", "diag_h_min", "diag_h_max",
            "fraction_logits_clipped_upper", "fraction_logits_clipped_lower",
            "fraction_rows_tied_max", "mean_tied_max", "max_tied_max", "coef_l2"]
    for (transform, dataset), g in all_results.groupby(["transform", "dataset"]):
        for label, keys in recipes.items():
            row = select(g, **keys)
            if row is not None:
                item = {"transform": transform, "dataset": dataset, "configuration": label}
                item.update({column: row.get(column, np.nan) for column in keep})
                headline.append(item)
    pd.DataFrame(headline).to_csv(out / "headline_configuration_comparison.csv", index=False)

    # Prediction-rule pairs share exactly the same fitted ridge and calibration.
    pairs = []
    pair_keys = ["transform", "dataset", "centered", "dtype",
                 "calibration_constraint", "lambda_source"]
    preval = all_results[all_results.model == "DiagnosticPreVal"]
    for keys, g in preval.groupby(pair_keys):
        raw = select(g, decision_rule="raw_score")
        prob = select(g, decision_rule="clipped_softmax")
        if raw is not None and prob is not None:
            item = dict(zip(pair_keys, keys))
            item.update(raw_accuracy=raw.accuracy, clipped_softmax_accuracy=prob.accuracy,
                        raw_minus_clipped_accuracy=raw.accuracy-prob.accuracy,
                        fraction_rows_tied_max=prob.fraction_rows_tied_max,
                        fraction_logits_clipped_upper=prob.fraction_logits_clipped_upper,
                        fraction_logits_clipped_lower=prob.fraction_logits_clipped_lower)
            pairs.append(item)
    pd.DataFrame(pairs).to_csv(out / "decision_rule_pairs.csv", index=False)

    # Saved class predictions permit exact agreement checks with RidgeCV.
    agreements = []
    for path in root.glob("*/*/predictions.npz"):
        with np.load(path, allow_pickle=False) as pred:
            for centered in (False, True):
                reference = pred[f"RidgeCV_centered={centered}"]
                for dtype in ("float32", "float64"):
                    for rule in ("raw_score", "clipped_softmax"):
                        key = f"{centered}|{dtype}|unconstrained|preval|{rule}"
                        candidate = pred[key]
                        agreements.append({
                            "transform": path.parent.parent.name,
                            "dataset": path.parent.name,
                            "centered": centered,
                            "dtype": dtype,
                            "decision_rule": rule,
                            "agreement_with_matching_ridgecv": float(np.mean(candidate == reference)),
                        })
    pd.DataFrame(agreements).to_csv(out / "prediction_agreement_with_ridgecv.csv", index=False)
    completed = all_results[["transform", "dataset"]].drop_duplicates().sort_values(["transform", "dataset"])
    completed.to_csv(out / "completed_cases.csv", index=False)
    print(f"Combined {len(completed)} cases and {len(all_results)} configurations")


if __name__ == "__main__":
    main()
