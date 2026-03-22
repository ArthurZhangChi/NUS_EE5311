import os
import json
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILES = [
    "B0005_soc_data.csv",
    "B0006_soc_data.csv",
    "B0007_soc_data.csv",
    "B0018_soc_data.csv",
]
# ============================================================
# 1. Basic utilities
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def metrics_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(title: str, y_true, y_pred):
    m = metrics_dict(y_true, y_pred)
    print(f"{title}: MAE={m['MAE']:.6f}, RMSE={m['RMSE']:.6f}, R2={m['R2']:.6f}")
    return m


def timed_predict(model, X):
    """
    Time one full-batch prediction call.
    Returns:
      y_pred_raw: raw model prediction
      total_time_sec: total prediction time in seconds
      avg_time_ms_per_sample: average inference time per sample in milliseconds
    """
    t0 = time.perf_counter()
    y_pred_raw = model.predict(X)
    t1 = time.perf_counter()

    total_time_sec = t1 - t0
    n_samples = len(X)
    avg_time_ms_per_sample = (total_time_sec / n_samples) * 1000.0 if n_samples > 0 else np.nan

    return y_pred_raw, total_time_sec, avg_time_ms_per_sample


# ============================================================
# 2. Load and validate CSV
# ============================================================

REQUIRED_COLS = [
    "battery_id",
    "cycle_number",
    "time",
    "voltage_measured",
    "current",
    "temperature",
    "soc",
    "capacity",
    "current_load",
    "voltage_load",
]

FEATURE_COLS = [
    "voltage_measured",
    "current",
    "temperature",
    "capacity",
    "current_load",
    "voltage_load",
]

TARGET_COL = "soc"


def load_battery_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLS).copy()

    # enforce numeric where appropriate
    numeric_cols = [
        "cycle_number",
        "time",
        "voltage_measured",
        "current",
        "temperature",
        "soc",
        "capacity",
        "current_load",
        "voltage_load",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS).copy()

    # enforce sort
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    # basic checks
    if not ((df["soc"] >= 0).all() and (df["soc"] <= 1).all()):
        print("[WARN] SOC is not fully within [0, 1]. Please confirm label scale.")

    return df


# ============================================================
# 3. Split by cycle: 6/2/2
# ============================================================

def split_cycles_6_2_2(df: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    cycles = sorted(df["cycle_number"].unique().tolist())
    n = len(cycles)

    if n < 5:
        raise ValueError(f"Too few cycles ({n}) to do a stable 6/2/2 split.")

    n_train = int(np.floor(n * 0.6))
    n_val = int(np.floor(n * 0.2))
    n_test = n - n_train - n_val

    train_cycles = cycles[:n_train]
    val_cycles = cycles[n_train:n_train + n_val]
    test_cycles = cycles[n_train + n_val:]

    if len(test_cycles) == 0:
        raise ValueError("Test split is empty. Please check cycle count.")

    return train_cycles, val_cycles, test_cycles


def subset_by_cycles(df: pd.DataFrame, cycles: List[int]) -> pd.DataFrame:
    out = df[df["cycle_number"].isin(cycles)].copy()
    out = out.sort_values(["cycle_number", "time"]).reset_index(drop=True)
    return out


# ============================================================
# 4. Current-timestep sample construction
# ============================================================

def make_instant_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build one sample per row using current-timestep features only.
    Output:
      X shape = [N, num_features]
      y shape = [N]
      meta contains battery_id, cycle_number, time
    """
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    meta = df[["battery_id", "cycle_number", "time"]].copy()

    if len(X) == 0:
        raise ValueError("No samples were created after preprocessing.")

    return X, y, meta


# ============================================================
# 5. Models
# ============================================================

def build_models(random_state: int = 42):
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )

    return {
        "RandomForest": rf,
        "MLP": mlp,
    }


# ============================================================
# 6. Visualization
# ============================================================

def plot_example_cycle(pred_df: pd.DataFrame, model_name: str, outdir: str, cycle_number=None):
    """
    Plot one test cycle: true SOC vs predicted SOC

    Parameters
    ----------
    pred_df : DataFrame
        Must contain columns: cycle_number, time, y_true, y_pred
    model_name : str
        Model name for plot title/file name
    outdir : str
        Output directory
    cycle_number : int or None
        If provided, plot this specific cycle.
        If None, use the middle cycle automatically.
    """
    if pred_df.empty:
        print(f"[WARN] {model_name}: pred_df is empty, skip plotting.")
        return

    available_cycles = sorted(pred_df["cycle_number"].unique().tolist())

    if cycle_number is None:
        chosen_cycle = available_cycles[len(available_cycles) // 2]
    else:
        if cycle_number not in available_cycles:
            print(f"[WARN] {model_name}: cycle {cycle_number} not found in test predictions.")
            print(f"[INFO] Available test cycles: {available_cycles}")
            return
        chosen_cycle = cycle_number

    g = pred_df[pred_df["cycle_number"] == chosen_cycle].copy()
    g = g.sort_values("time")

    plt.figure(figsize=(10, 5))
    plt.plot(g["time"], g["y_true"], label="True SOC")
    plt.plot(g["time"], g["y_pred"], label="Predicted SOC")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.title(f"{model_name} - Test Cycle {chosen_cycle}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_cycle_{chosen_cycle}.png"), dpi=150)
    plt.close()

    print(f"[INFO] Saved example cycle plot: {model_name}_cycle_{chosen_cycle}.png")


def plot_scatter(pred_df: pd.DataFrame, model_name: str, outdir: str):
    if pred_df.empty:
        return

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(pred_df["y_true"], pred_df["y_pred"], s=8, alpha=0.5)
    mn = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    mx = max(pred_df["y_true"].max(), pred_df["y_pred"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True SOC")
    plt.ylabel("Predicted SOC")
    plt.title(f"{model_name} - True vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_scatter.png"), dpi=150)
    plt.close()


# ============================================================
# 7. Main experiment
# ============================================================

def run_experiment(csv_path: str, window_size: int, outdir: str, random_state: int = 42):
    ensure_dir(outdir)

    df = load_battery_csv(csv_path)
    battery_ids = df["battery_id"].unique().tolist()

    if len(battery_ids) != 1:
        print(f"[WARN] This CSV contains multiple battery_id values: {battery_ids}")
    battery_name = str(battery_ids[0])

    print(f"[INFO] Loaded {csv_path}")
    print(f"[INFO] Battery: {battery_name}")
    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Total cycles: {df['cycle_number'].nunique()}")

    # split by cycle
    train_cycles, val_cycles, test_cycles = split_cycles_6_2_2(df)

    train_df = subset_by_cycles(df, train_cycles)
    val_df = subset_by_cycles(df, val_cycles)
    test_df = subset_by_cycles(df, test_cycles)

    print(f"[INFO] Train cycles: {len(train_cycles)} | rows={len(train_df)}")
    print(f"[INFO] Val cycles:   {len(val_cycles)} | rows={len(val_df)}")
    print(f"[INFO] Test cycles:  {len(test_cycles)} | rows={len(test_df)}")

    # build current-timestep samples
    X_train, y_train, meta_train = make_instant_samples(
        train_df, FEATURE_COLS, TARGET_COL
    )
    X_val, y_val, meta_val = make_instant_samples(
        val_df, FEATURE_COLS, TARGET_COL
    )
    X_test, y_test, meta_test = make_instant_samples(
        test_df, FEATURE_COLS, TARGET_COL
    )

    print(f"[INFO] Sample mode: current_timestep_only")
    print(f"[INFO] Note: --window={window_size} is ignored in this version")
    print(f"[INFO] Samples -> train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # scaler fit on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = build_models(random_state=random_state)
    summary_rows = []

    # save protocol
    protocol = {
        "csv_path": csv_path,
        "battery_id": battery_name,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "sample_mode": "current_timestep_only",
        "window_size": None,
        "split_rule": "cycle-wise 6/2/2 in chronological order",
        "train_cycles": train_cycles,
        "val_cycles": val_cycles,
        "test_cycles": test_cycles,
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "n_test_samples": int(len(X_test)),
    }
    with open(os.path.join(outdir, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)

    for model_name, model in models.items():
        print("\n" + "=" * 60)
        print(f"[INFO] Training model: {model_name}")

        # train
        model.fit(X_train_scaled, y_train)

        # validation prediction
        y_val_pred_raw, val_total_time_sec, val_avg_ms_per_sample = timed_predict(model, X_val_scaled)
        y_val_pred = np.clip(y_val_pred_raw, 0.0, 1.0)

        # test prediction
        y_test_pred_raw, test_total_time_sec, test_avg_ms_per_sample = timed_predict(model, X_test_scaled)
        y_test_pred = np.clip(y_test_pred_raw, 0.0, 1.0)

        val_metrics = print_metrics(f"{model_name} | VAL", y_val, y_val_pred)
        test_metrics = print_metrics(f"{model_name} | TEST", y_test, y_test_pred)

        print(
            f"{model_name} | VAL inference: total={val_total_time_sec * 1000:.4f} ms, "
            f"avg={val_avg_ms_per_sample:.6f} ms/sample"
        )
        print(
            f"{model_name} | TEST inference: total={test_total_time_sec * 1000:.4f} ms, "
            f"avg={test_avg_ms_per_sample:.6f} ms/sample"
        )

        pred_df = meta_test.copy()
        pred_df["y_true"] = y_test
        pred_df["y_pred"] = y_test_pred
        pred_df.to_csv(
            os.path.join(outdir, f"{model_name}_test_predictions.csv"),
            index=False
        )

        plot_example_cycle(pred_df, model_name, outdir, cycle_number=135)
        plot_scatter(pred_df, model_name, outdir)

        row = {
            "battery_id": battery_name,
            "model": model_name,
            "sample_mode": "current_timestep_only",

            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],

            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],

            "val_total_inference_time_ms": float(val_total_time_sec * 1000.0),
            "val_avg_inference_time_ms_per_sample": float(val_avg_ms_per_sample),

            "test_total_inference_time_ms": float(test_total_time_sec * 1000.0),
            "test_avg_inference_time_ms_per_sample": float(test_avg_ms_per_sample),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("[INFO] Finished.")
    print(summary_df)


# ============================================================
# 8. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=20, help="Kept only for CLI compatibility; ignored")
    parser.add_argument("--outdir", type=str, default="results_soc", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for models")
    args = parser.parse_args()

    run_experiment(
        csv_path="B0018_soc_data.csv",
        window_size=args.window,
        outdir=args.outdir,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()