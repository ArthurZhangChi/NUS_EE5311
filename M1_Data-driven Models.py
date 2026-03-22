from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
FEATURE_COLS = [
    "voltage_measured",
    "current",
    "temperature",
    "capacity",
    "current_load",
    "voltage_load",
]
TARGET_COL = "soc"
SOC_MIN = 0.0
SOC_MAX = 1.0
DISPLAY_NAME_MAP = {
    "RandomForest": "RandomForest",
    "TorchMLP_Matched": "MLP",
}
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


# ============================================================
# Utilities
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested_device


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = metrics_dict(y_true, y_pred)
    print(
        f"{title}: "
        f"MAE={metrics['MAE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}, "
        f"R2={metrics['R2']:.6f}"
    )
    return metrics


def timed_predict_sklearn(model: Any, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    y_pred_raw = model.predict(x)
    t1 = time.perf_counter()

    total_time_sec = t1 - t0
    n_samples = len(x)
    avg_time_ms = (total_time_sec / n_samples) * 1000.0 if n_samples > 0 else np.nan
    return y_pred_raw, total_time_sec, avg_time_ms


def timed_predict_torch(model: nn.Module, x: np.ndarray, device: str) -> Tuple[np.ndarray, float, float]:
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        y_pred_raw = model(x_t)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    y_pred_raw = y_pred_raw.detach().cpu().numpy()
    total_time_sec = t1 - t0
    n_samples = len(x)
    avg_time_ms = (total_time_sec / n_samples) * 1000.0 if n_samples > 0 else np.nan
    return y_pred_raw, total_time_sec, avg_time_ms


# ============================================================
# Data loading and preprocessing
# ============================================================


def resolve_dataset_path(dataset_dir: Path, battery_id: str) -> Path:
    """
    Find the CSV for one battery inside datasets/.
    This function is intentionally tolerant to minor filename differences,
    as long as the filename contains the battery ID.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    candidates = sorted(dataset_dir.glob("*.csv"))
    exact_matches = [p for p in candidates if p.stem.upper() == battery_id.upper()]
    if exact_matches:
        return exact_matches[0]

    contains_matches = [p for p in candidates if battery_id.upper() in p.stem.upper()]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1:
        names = ", ".join(str(p.name) for p in contains_matches)
        raise FileExistsError(f"Multiple CSV files match {battery_id}: {names}")

    available = ", ".join(p.name for p in candidates) if candidates else "<empty directory>"
    raise FileNotFoundError(
        f"Could not find a CSV for {battery_id} in {dataset_dir}. Available files: {available}"
    )



def load_battery_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    df = df.dropna(subset=REQUIRED_COLS).copy()

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
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    if not ((df[TARGET_COL] >= SOC_MIN).all() and (df[TARGET_COL] <= SOC_MAX).all()):
        print(f"[WARN] SOC in {csv_path.name} is not fully within [0, 1].")

    return df



def split_cycles_6_2_2(df: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    cycles = sorted(df["cycle_number"].unique().tolist())
    n_cycles = len(cycles)

    if n_cycles < 5:
        raise ValueError(f"Too few cycles ({n_cycles}) to perform a stable 6/2/2 split.")

    n_train = int(np.floor(n_cycles * 0.6))
    n_val = int(np.floor(n_cycles * 0.2))
    n_test = n_cycles - n_train - n_val

    train_cycles = cycles[:n_train]
    val_cycles = cycles[n_train:n_train + n_val]
    test_cycles = cycles[n_train + n_val:]

    if n_test <= 0:
        raise ValueError("Test split is empty. Please check cycle count.")

    return train_cycles, val_cycles, test_cycles



def subset_by_cycles(df: pd.DataFrame, cycles: List[int]) -> pd.DataFrame:
    out = df[df["cycle_number"].isin(cycles)].copy()
    return out.sort_values(["cycle_number", "time"]).reset_index(drop=True)



def make_instant_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    x = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    meta = df[["battery_id", "cycle_number", "time"]].copy()

    if len(x) == 0:
        raise ValueError("No samples were created after preprocessing.")

    return x, y, meta


# ============================================================
# Models
# ============================================================


class TorchMLPMatched(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)



def build_random_forest(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )



def train_torch_model_matched(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 30,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_mse": [], "val_mse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train = model(x_train_t)
        train_mse = torch.mean((pred_train - y_train_t) ** 2)
        train_mse.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_mse = torch.mean((pred_val - y_val_t) ** 2).item()

        history["train_mse"].append(float(train_mse.item()))
        history["val_mse"].append(float(val_mse))

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"train_mse={train_mse.item():.6f} "
                f"val_mse={val_mse:.6f}"
            )

        if wait >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ============================================================
# Plotting
# ============================================================


def load_first_test_cycle(result_dir: Path, model_name: str) -> Dict[str, Any]:
    pred_file = result_dir / f"{model_name}_test_predictions.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    df = pd.read_csv(pred_file)
    required_cols = ["cycle_number", "y_true", "y_pred"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {pred_file.name}: {missing}")

    if "time" in df.columns:
        df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)
    else:
        df = df.sort_values(["cycle_number"]).reset_index(drop=True)

    first_cycle = sorted(df["cycle_number"].unique().tolist())[0]
    cycle_df = df[df["cycle_number"] == first_cycle].copy()
    cycle_df = cycle_df.rename(columns={"y_true": "soc_true", "y_pred": "soc_pred"})

    return {
        "dataset_name": result_dir.name.replace("results_", ""),
        "first_cycle": first_cycle,
        "cycle_df": cycle_df,
    }



def plot_first_test_cycle_4subplots(
    plot_data_list: List[Dict[str, Any]],
    save_path: Path,
    model_name: str,
) -> None:
    display_name = DISPLAY_NAME_MAP.get(model_name, model_name)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, item in zip(axes, plot_data_list):
        dataset_name = item["dataset_name"]
        first_cycle = item["first_cycle"]
        cycle_df = item["cycle_df"]

        x = np.arange(len(cycle_df))
        y_true = cycle_df["soc_true"].values
        y_pred = cycle_df["soc_pred"].values

        ax.plot(x, y_true, label="Actual SOC", linewidth=1.8)
        ax.plot(x, y_pred, label="Predicted SOC", linewidth=1.8)
        ax.set_title(f"{dataset_name} - First Test Cycle ({first_cycle}) [{display_name}]")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("SOC")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for i in range(len(plot_data_list), 4):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved combined first-test-cycle plot: {save_path}")


# ============================================================
# Single-battery experiment
# ============================================================


def run_single_battery_experiment(
    csv_path: Path,
    outdir: Path,
    random_state: int = 42,
    device: str = "cpu",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    patience: int = 30,
) -> pd.DataFrame:
    ensure_dir(outdir)
    set_seed(random_state)
    df = load_battery_csv(csv_path)
    battery_ids = df["battery_id"].unique().tolist()
    if len(battery_ids) != 1:
        print(f"[WARN] This CSV contains multiple battery_id values: {battery_ids}")
    battery_name = str(battery_ids[0])

    print("\n" + "=" * 72)
    print(f"[INFO] Loaded {csv_path}")
    print(f"[INFO] Battery: {battery_name}")
    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Total cycles: {df['cycle_number'].nunique()}")

    train_cycles, val_cycles, test_cycles = split_cycles_6_2_2(df)
    train_df = subset_by_cycles(df, train_cycles)
    val_df = subset_by_cycles(df, val_cycles)
    test_df = subset_by_cycles(df, test_cycles)

    print(f"[INFO] Train cycles: {len(train_cycles)} | rows={len(train_df)}")
    print(f"[INFO] Val cycles:   {len(val_cycles)} | rows={len(val_df)}")
    print(f"[INFO] Test cycles:  {len(test_cycles)} | rows={len(test_df)}")

    x_train, y_train, _ = make_instant_samples(train_df, FEATURE_COLS, TARGET_COL)
    x_val, y_val, _ = make_instant_samples(val_df, FEATURE_COLS, TARGET_COL)
    x_test, y_test, meta_test = make_instant_samples(test_df, FEATURE_COLS, TARGET_COL)

    print(f"[INFO] Samples -> train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_val_scaled = scaler.transform(x_val).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    summary_rows: List[Dict[str, Any]] = []

    # ---------------------------
    # RandomForest
    # ---------------------------
    rf_name = "RandomForest"
    print("\n" + "-" * 72)
    print(f"[INFO] Training model: {rf_name}")
    rf_model = build_random_forest(random_state=random_state)
    rf_model.fit(x_train_scaled, y_train)

    y_val_pred_raw, val_total_sec, val_avg_ms = timed_predict_sklearn(rf_model, x_val_scaled)
    y_test_pred_raw, test_total_sec, test_avg_ms = timed_predict_sklearn(rf_model, x_test_scaled)
    y_val_pred = np.clip(y_val_pred_raw, SOC_MIN, SOC_MAX)
    y_test_pred = np.clip(y_test_pred_raw, SOC_MIN, SOC_MAX)

    val_metrics = print_metrics(f"{rf_name} | VAL", y_val, y_val_pred)
    test_metrics = print_metrics(f"{rf_name} | TEST", y_test, y_test_pred)

    rf_pred_df = meta_test.copy()
    rf_pred_df["y_true"] = y_test
    rf_pred_df["y_pred"] = y_test_pred
    rf_pred_df.to_csv(outdir / f"{rf_name}_test_predictions.csv", index=False)

    summary_rows.append(
        {
            "battery_id": battery_name,
            "model": rf_name,
            "sample_mode": "current_timestep_only",
            "train_method": "sklearn_default",
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "val_total_inference_time_ms": float(val_total_sec * 1000.0),
            "val_avg_inference_time_ms_per_sample": float(val_avg_ms),
            "test_total_inference_time_ms": float(test_total_sec * 1000.0),
            "test_avg_inference_time_ms_per_sample": float(test_avg_ms),
            "epochs_ran": np.nan,
            "best_val_mse": np.nan,
        }
    )

    # ---------------------------
    # TorchMLP_Matched
    # ---------------------------
    mlp_name = "TorchMLP_Matched"
    print("\n" + "-" * 72)
    print(f"[INFO] Training model: {mlp_name}")

    torch_model = TorchMLPMatched(input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim).to(device)
    torch_model, history = train_torch_model_matched(
        model=torch_model,
        x_train=x_train_scaled,
        y_train=y_train,
        x_val=x_val_scaled,
        y_val=y_val,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    y_val_pred_raw, val_total_sec, val_avg_ms = timed_predict_torch(torch_model, x_val_scaled, device=device)
    y_test_pred_raw, test_total_sec, test_avg_ms = timed_predict_torch(torch_model, x_test_scaled, device=device)
    y_val_pred = np.clip(y_val_pred_raw, SOC_MIN, SOC_MAX)
    y_test_pred = np.clip(y_test_pred_raw, SOC_MIN, SOC_MAX)

    val_metrics = print_metrics(f"{mlp_name} | VAL", y_val, y_val_pred)
    test_metrics = print_metrics(f"{mlp_name} | TEST", y_test, y_test_pred)

    mlp_pred_df = meta_test.copy()
    mlp_pred_df["y_true"] = y_test
    mlp_pred_df["y_pred"] = y_test_pred
    mlp_pred_df.to_csv(outdir / f"{mlp_name}_test_predictions.csv", index=False)

    summary_rows.append(
        {
            "battery_id": battery_name,
            "model": mlp_name,
            "sample_mode": "current_timestep_only",
            "train_method": "full-batch_adam_external-val_early-stopping",
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "val_total_inference_time_ms": float(val_total_sec * 1000.0),
            "val_avg_inference_time_ms_per_sample": float(val_avg_ms),
            "test_total_inference_time_ms": float(test_total_sec * 1000.0),
            "test_avg_inference_time_ms_per_sample": float(test_avg_ms),
            "epochs_ran": int(len(history["val_mse"])),
            "best_val_mse": float(min(history["val_mse"])) if history["val_mse"] else np.nan,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "metrics_summary.csv", index=False)
    print(f"[INFO] Saved summary: {outdir / 'metrics_summary.csv'}")

    return summary_df


# ============================================================
# Full experiment across four batteries
# ============================================================


def run_all_batteries(
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    device: str = "cpu",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim: int = 64,
    patience: int = 30,
) -> None:
    ensure_dir(output_dir)
    set_seed(seed)
    device = resolve_device(device)

    all_summary_frames: List[pd.DataFrame] = []
    result_dirs: List[Path] = []

    for battery_id in BATTERY_IDS:
        csv_path = resolve_dataset_path(dataset_dir, battery_id)
        battery_outdir = output_dir / f"results_{battery_id}"
        result_dirs.append(battery_outdir)

        summary_df = run_single_battery_experiment(
            csv_path=csv_path,
            outdir=battery_outdir,
            random_state=seed,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            patience=patience,
        )
        all_summary_frames.append(summary_df)

    combined_summary = pd.concat(all_summary_frames, ignore_index=True)
    combined_summary.to_csv(output_dir / "all_metrics_summary.csv", index=False)
    print(f"\n[INFO] Saved combined summary: {output_dir / 'all_metrics_summary.csv'}")

    rf_plot_data = [load_first_test_cycle(result_dir, "RandomForest") for result_dir in result_dirs]
    mlp_plot_data = [load_first_test_cycle(result_dir, "TorchMLP_Matched") for result_dir in result_dirs]

    plot_first_test_cycle_4subplots(
        plot_data_list=rf_plot_data,
        save_path=output_dir / "rf_first_test_cycle_4subplots.png",
        model_name="RandomForest",
    )
    plot_first_test_cycle_4subplots(
        plot_data_list=mlp_plot_data,
        save_path=output_dir / "torchmlp_matched_first_test_cycle_4subplots.png",
        model_name="TorchMLP_Matched",
    )

    print("\n[INFO] Finished running all four datasets.")
    print(f"[INFO] Outputs saved under: {output_dir}")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run SOC experiments for B0005, B0006, B0007, and B0018 using only "
            "RandomForest and TorchMLP_Matched, then generate combined first-test-cycle plots."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=script_dir / "datasets",
        help="Directory containing the four battery CSV files. Default: ./datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=script_dir / "outputs",
        help="Directory for all experiment outputs. Default: ./outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for TorchMLP_Matched",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Epochs for TorchMLP_Matched")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for TorchMLP_Matched")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for TorchMLP_Matched",
    )
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for TorchMLP_Matched")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    args = parser.parse_args()

    run_all_batteries(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
