"""Combine completed case results and compute one-factor intervention contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    completed = all_results[["transform", "dataset"]].drop_duplicates().sort_values(["transform", "dataset"])
    completed.to_csv(out / "completed_cases.csv", index=False)
    print(f"Combined {len(completed)} cases and {len(all_results)} configurations")


if __name__ == "__main__":
    main()
