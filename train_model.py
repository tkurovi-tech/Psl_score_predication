"""
train_model.py
==============
PSL 1st Innings Score Predictor – Model Training Script
SZABIST · Department of Robotics & Artificial Intelligence

Usage:
    python train_model.py --data PSL_1st_Inning_Clean.csv

Outputs (saved to same directory):
    model.pkl        – Trained Random Forest model
    le_venue.pkl     – LabelEncoder for venue
    le_team.pkl      – LabelEncoder for team
    le_opp.pkl       – LabelEncoder for opposition
    training_report.txt – Performance report
"""

import argparse
import os
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Config ────────────────────────────────────────────────────────────────────
FEATURES = [
    "venue_enc", "team_enc", "opp_enc",
    "Over Number", "Ball Number",
    "current_run", "run_rate",
    "wickets_left", "runs_till_5_over",
]
TARGET = "score"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ── Load & Clean ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading data from: {path}")
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()], inplace=True)

    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"      Rows: {before:,}  →  {after:,} after dropping nulls")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    print("[2/5] Encoding categorical features …")

    le_venue = LabelEncoder()
    le_team  = LabelEncoder()
    le_opp   = LabelEncoder()

    df["venue_enc"] = le_venue.fit_transform(df["venue"])
    df["team_enc"]  = le_team.fit_transform(df["Team"])
    df["opp_enc"]   = le_opp.fit_transform(df["Opposition"])

    print(f"      Venues  : {len(le_venue.classes_)}")
    print(f"      Teams   : {len(le_team.classes_)}")

    return df, le_venue, le_team, le_opp


# ── Train ─────────────────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    print("[3/5] Splitting data and training models …")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    candidates = {
        "Random Forest":      RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=150, random_state=RANDOM_STATE),
        "Ridge Regression":   Ridge(),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = mean_squared_error(y_test, preds) ** 0.5
        r2    = r2_score(y_test, preds)
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}
        print(f"      {name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    # Pick best by R²
    best_name  = max(results, key=lambda n: results[n]["r2"])
    best       = results[best_name]
    best_model = best["model"]
    print(f"\n      ✅ Best model: {best_name}  (R²={best['r2']:.4f})")

    return best_model, best_name, results, X_test, y_test


# ── Save Artifacts ─────────────────────────────────────────────────────────────
def save_artifacts(model, le_venue, le_team, le_opp, out_dir: str):
    print(f"[4/5] Saving model artifacts to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    def _save(obj, name):
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"      Saved → {name}")

    _save(model,    "model.pkl")
    _save(le_venue, "le_venue.pkl")
    _save(le_team,  "le_team.pkl")
    _save(le_opp,   "le_opp.pkl")


# ── Report & Plots ────────────────────────────────────────────────────────────
def generate_report(best_name, results, model, X_test, y_test, out_dir: str):
    print("[5/5] Generating training report and plots …")
    os.makedirs(out_dir, exist_ok=True)

    # ── Text report ──
    lines = [
        "=" * 60,
        "  PSL 1st Innings Score Predictor – Training Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60, "",
        "Model Comparison:",
        "-" * 60,
    ]
    for name, r in results.items():
        mark = " ← SELECTED" if name == best_name else ""
        lines += [
            f"  {name}{mark}",
            f"    MAE  : {r['mae']:.2f} runs",
            f"    RMSE : {r['rmse']:.2f} runs",
            f"    R²   : {r['r2']:.4f}",
            "",
        ]
    lines += ["Feature Importance (Random Forest):", "-" * 60]

    if hasattr(model, "feature_importances_"):
        imp = sorted(
            zip(FEATURES, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for feat, val in imp:
            lines.append(f"  {feat:25s}  {val:.4f}")

    report_path = os.path.join(out_dir, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"      Report → training_report.txt")

    # ── Actual vs Predicted scatter ──
    preds = model.predict(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.scatter(y_test, preds, alpha=.3, color="#00e676", s=8)
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Score", color="white")
    ax.set_ylabel("Predicted Score", color="white")
    ax.set_title("Actual vs Predicted", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#0d1117", labelcolor="white")

    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    residuals = preds - y_test
    ax2.hist(residuals, bins=40, color="#388bfd", edgecolor="#0d1117", alpha=.8)
    ax2.axvline(0, color="#f85149", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Residual (Predicted − Actual)", color="white")
    ax2.set_ylabel("Frequency", color="white")
    ax2.set_title("Residual Distribution", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values(): spine.set_edgecolor("#30363d")

    plt.suptitle(f"{best_name} Performance", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "model_performance.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"      Plot  → model_performance.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train PSL Score Predictor")
    parser.add_argument("--data",    default="PSL_1st_Inning_Clean.csv", help="Path to CSV dataset")
    parser.add_argument("--out-dir", default=".",                         help="Output directory for artifacts")
    args = parser.parse_args()

    df                          = load_data(args.data)
    df, le_venue, le_team, le_opp = engineer_features(df)
    model, best_name, results, X_test, y_test = train_and_evaluate(df)
    save_artifacts(model, le_venue, le_team, le_opp, args.out_dir)
    generate_report(best_name, results, model, X_test, y_test, args.out_dir)

    print("\n✅  Done! Run the web app with:  python app.py")


if __name__ == "__main__":
    main()
