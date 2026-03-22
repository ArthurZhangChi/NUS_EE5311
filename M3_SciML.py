import os
import random
import time
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Global config
# ============================================================
SEED = 0
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DEVICE = "cpu"

CSV_FILES = [
    "datasets/B0005_soc_data.csv",
    "datasets/B0006_soc_data.csv",
    "datasets/B0007_soc_data.csv",
    "datasets/B0018_soc_data.csv",
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

LAMBDA_PHYS = 0.2
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_DIM = 64
PATIENCE = 30

SOC_MIN = 0.0
SOC_MAX = 1.0

OUTPUT_SUMMARY_CSV = "M3_test_predictions.csv"
OUTPUT_METRICS_ONLY_CSV = "M3_metrics.csv"
OUTPUT_FIGURE = "M3_first_test_cycle_4subplots.png"


# ============================================================
# 2. Reproducibility
# ============================================================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Data loading
# ============================================================
def load_single_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = [
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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df = df.copy()
    df["source_file"] = os.path.basename(csv_path)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(f"{csv_path} has too few valid rows after dropping NaNs: {len(df)}")

    return df


# ============================================================
# 4. Split by ordered cycle_number
# ============================================================
def split_by_cycle_number(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    if "cycle_number" not in df.columns:
        raise ValueError("DataFrame must contain 'cycle_number' column.")

    unique_cycles = sorted(df["cycle_number"].dropna().unique().tolist())
    n_cycles = len(unique_cycles)

    if n_cycles < 3:
        raise ValueError(f"Not enough unique cycles to split: {n_cycles}")

    n_train_cycles = int(n_cycles * train_ratio)
    n_val_cycles = int(n_cycles * val_ratio)
    n_test_cycles = n_cycles - n_train_cycles - n_val_cycles

    if n_train_cycles == 0 or n_val_cycles == 0 or n_test_cycles == 0:
        raise ValueError(
            f"Cycle split produced empty subset: total_cycles={n_cycles}, "
            f"train_cycles={n_train_cycles}, val_cycles={n_val_cycles}, test_cycles={n_test_cycles}"
        )

    train_cycles = unique_cycles[:n_train_cycles]
    val_cycles = unique_cycles[n_train_cycles:n_train_cycles + n_val_cycles]
    test_cycles = unique_cycles[n_train_cycles + n_val_cycles:]

    train_df = df[df["cycle_number"].isin(train_cycles)].copy()
    val_df = df[df["cycle_number"].isin(val_cycles)].copy()
    test_df = df[df["cycle_number"].isin(test_cycles)].copy()

    sort_cols = ["battery_id", "cycle_number", "time"]
    train_df = train_df.sort_values(sort_cols).reset_index(drop=True)
    val_df = val_df.sort_values(sort_cols).reset_index(drop=True)
    test_df = test_df.sort_values(sort_cols).reset_index(drop=True)

    return train_df, val_df, test_df


# ============================================================
# 5. Model
# ============================================================
class SciMLSOCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = torch.sigmoid(self.head(x))
        return x.squeeze(-1)


# ============================================================
# 6. Utility
# ============================================================
def prepare_xy(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    x = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    else:
        if scaler is None:
            raise ValueError("Scaler must not be None when fit_scaler=False.")
        x = scaler.transform(x)

    return x, y, scaler


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return float(rmse), float(mae), float(r2)


# ============================================================
# 7. Loss functions
# ============================================================
def compute_data_loss(pred_soc: torch.Tensor, true_soc: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred_soc - true_soc) ** 2)


def compute_physics_loss(df: pd.DataFrame, pred_soc_np: np.ndarray) -> float:
    tmp = df.copy()
    tmp["pred_soc"] = pred_soc_np
    tmp = tmp.sort_values(["battery_id", "cycle_number", "time"]).reset_index(drop=True)

    losses: List[float] = []

    grouped = tmp.groupby(["battery_id", "cycle_number"], sort=False)
    for _, g in grouped:
        if len(g) < 2:
            continue

        t = g["time"].values.astype(np.float64)
        i = -g["current"].values.astype(np.float64)
        c = g["capacity"].values.astype(np.float64)
        s_pred = g["pred_soc"].values.astype(np.float64)

        dt = np.diff(t)
        dsoc_pred = np.diff(s_pred)

        i_avg = 0.5 * (i[:-1] + i[1:])
        c_avg = 0.5 * (c[:-1] + c[1:])

        c_avg = np.clip(c_avg, 1e-6, None)
        dsoc_phys = -(i_avg * dt) / (c_avg * 3600.0)

        valid = np.isfinite(dt) & np.isfinite(dsoc_pred) & np.isfinite(dsoc_phys)
        valid &= (np.abs(dt) > 1e-12)

        if np.any(valid):
            loss = np.mean((dsoc_pred[valid] - dsoc_phys[valid]) ** 2)
            losses.append(float(loss))

    if len(losses) == 0:
        return 0.0

    return float(np.mean(losses))


def compute_physics_loss_torch(
    df: pd.DataFrame,
    pred_soc: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    if pred_soc.ndim != 1:
        pred_soc = pred_soc.view(-1)

    n = len(df)
    if pred_soc.shape[0] != n:
        raise ValueError(
            f"pred_soc length {pred_soc.shape[0]} does not match df length {n}"
        )

    time_t = torch.tensor(df["time"].values, dtype=torch.float32, device=device)
    current_t = torch.tensor(df["current"].values, dtype=torch.float32, device=device)
    capacity_t = torch.tensor(df["capacity"].values, dtype=torch.float32, device=device)

    battery_ids = df["battery_id"].values
    cycle_ids = df["cycle_number"].values

    current_eff = -current_t

    losses = []
    start = 0

    for k in range(1, n + 1):
        same_group = (
            k < n
            and battery_ids[k] == battery_ids[start]
            and cycle_ids[k] == cycle_ids[start]
        )

        if same_group:
            continue

        if k - start >= 2:
            t_g = time_t[start:k]
            i_g = current_eff[start:k]
            c_g = capacity_t[start:k]
            s_g = pred_soc[start:k]

            dt = t_g[1:] - t_g[:-1]
            dsoc_pred = s_g[1:] - s_g[:-1]

            i_avg = 0.5 * (i_g[:-1] + i_g[1:])
            c_avg = 0.5 * (c_g[:-1] + c_g[1:])
            c_avg = torch.clamp(c_avg, min=1e-6)

            dsoc_phys = -(i_avg * dt) / (c_avg * 3600.0)

            valid = torch.isfinite(dt) & torch.isfinite(dsoc_pred) & torch.isfinite(dsoc_phys)
            valid = valid & (torch.abs(dt) > 1e-12)

            if valid.any():
                group_loss = torch.mean((dsoc_pred[valid] - dsoc_phys[valid]) ** 2)
                losses.append(group_loss)

        start = k

    if len(losses) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    return torch.stack(losses).mean()


# ============================================================
# 8. Train one battery-specific model
# ============================================================
def train_model(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scaler: StandardScaler,
    device: str = "cpu",
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    lambda_phys: float = 0.2,
    patience: int = 30,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train, y_train, _ = prepare_xy(train_df, scaler=scaler, fit_scaler=False)
    x_val, y_val, _ = prepare_xy(val_df, scaler=scaler, fit_scaler=False)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    history = {
        "train_total": [],
        "train_data": [],
        "train_phys": [],
        "val_mse": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train = model(x_train_t)
        data_loss = compute_data_loss(pred_train, y_train_t)

        phys_loss = compute_physics_loss_torch(
            train_df,
            pred_train,
            device=device,
        )

        total_loss = data_loss + lambda_phys * phys_loss
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x_val_t)
            val_mse = compute_data_loss(pred_val, y_val_t).item()

        history["train_total"].append(float(total_loss.item()))
        history["train_data"].append(float(data_loss.item()))
        history["train_phys"].append(float(phys_loss.item()))
        history["val_mse"].append(float(val_mse))

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{epochs}] "
                f"train_total={total_loss.item():.6f} "
                f"train_data={data_loss.item():.6f} "
                f"train_phys={phys_loss.item():.6f} "
                f"val_mse={val_mse:.6f}"
            )

        if wait >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ============================================================
# 9. Get plot data for first cycle in test set
# ============================================================
def get_first_test_cycle_plot_data(
    test_df: pd.DataFrame,
    model: nn.Module,
    scaler: StandardScaler,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    first_cycle = sorted(test_df["cycle_number"].unique().tolist())[0]
    cycle_df = test_df[test_df["cycle_number"] == first_cycle].copy()
    cycle_df = cycle_df.sort_values(["battery_id", "cycle_number", "time"]).reset_index(drop=True)

    x_cycle, y_cycle, _ = prepare_xy(cycle_df, scaler=scaler, fit_scaler=False)
    x_cycle_t = torch.tensor(x_cycle, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        y_cycle_pred = model(x_cycle_t)

    if device == "cuda":
        torch.cuda.synchronize()

    y_cycle_pred = y_cycle_pred.cpu().numpy()
    y_cycle_pred = np.clip(y_cycle_pred, SOC_MIN, SOC_MAX)

    return {
        "cycle_number": int(first_cycle),
        "time": cycle_df["time"].values.astype(np.float64),
        "actual_soc": y_cycle.astype(np.float64),
        "predicted_soc": y_cycle_pred.astype(np.float64),
    }


# ============================================================
# 10. Save one big figure with 4 subplots
# ============================================================
def save_first_test_cycle_figure(
    plot_infos: List[Dict[str, np.ndarray]],
    save_path: str = OUTPUT_FIGURE,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, info in zip(axes, plot_infos):
        time_vals = info["time"]
        actual_soc = info["actual_soc"]
        pred_soc = info["predicted_soc"]
        battery_name = info["battery_name"]
        cycle_number = info["cycle_number"]

        ax.plot(time_vals, actual_soc, label="Actual SOC", linewidth=2)
        ax.plot(time_vals, pred_soc, label="Predicted SOC", linewidth=2)

        ax.set_title(f"{battery_name} - First Test Cycle ({cycle_number})", fontsize=14)
        ax.set_xlabel("Cycle Time")
        ax.set_ylabel("SOC")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for i in range(len(plot_infos), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved 4-subplot figure to: {save_path}")


# ============================================================
# 11. Run one battery file
# ============================================================
def run_one_battery(csv_path: str):
    battery_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n================ Processing {battery_name} ================")

    df = load_single_csv(csv_path)

    train_df, val_df, test_df = split_by_cycle_number(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    train_cycles = sorted(train_df["cycle_number"].unique().tolist())
    val_cycles = sorted(val_df["cycle_number"].unique().tolist())
    test_cycles = sorted(test_df["cycle_number"].unique().tolist())

    print(
        f"[INFO] {battery_name}: total_samples={len(df)}, "
        f"train_samples={len(train_df)}, val_samples={len(val_df)}, test_samples={len(test_df)}"
    )
    print(
        f"[INFO] {battery_name}: "
        f"train_cycles={train_cycles[0]}~{train_cycles[-1]} ({len(train_cycles)} cycles), "
        f"val_cycles={val_cycles[0]}~{val_cycles[-1]} ({len(val_cycles)} cycles), "
        f"test_cycles={test_cycles[0]}~{test_cycles[-1]} ({len(test_cycles)} cycles)"
    )

    x_train = train_df[FEATURE_COLS].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_train)

    model = SciMLSOCNet(input_dim=len(FEATURE_COLS), hidden_dim=HIDDEN_DIM).to(DEVICE)

    model, history = train_model(
        model=model,
        train_df=train_df,
        val_df=val_df,
        scaler=scaler,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        lambda_phys=LAMBDA_PHYS,
        patience=PATIENCE,
    )

    # ---------------- validation ----------------
    x_val, y_val, _ = prepare_xy(val_df, scaler=scaler, fit_scaler=False)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val_t)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    y_val_pred = y_val_pred.cpu().numpy()
    y_val_pred = np.clip(y_val_pred, SOC_MIN, SOC_MAX)

    val_rmse, val_mae, val_r2 = evaluate_metrics(y_val, y_val_pred)
    val_phys_loss = compute_physics_loss(val_df, y_val_pred)

    # ---------------- test + inference time ----------------
    x_test, y_test, _ = prepare_xy(test_df, scaler=scaler, fit_scaler=False)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)

    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_t)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    test_inference_time_sec = end_time - start_time

    y_test_pred = y_test_pred.cpu().numpy()
    y_test_pred = np.clip(y_test_pred, SOC_MIN, SOC_MAX)

    test_rmse, test_mae, test_r2 = evaluate_metrics(y_test, y_test_pred)
    test_phys_loss = compute_physics_loss(test_df, y_test_pred)

    # ---------------- first test cycle plot data ----------------
    plot_info = get_first_test_cycle_plot_data(
        test_df=test_df,
        model=model,
        scaler=scaler,
        device=DEVICE,
    )
    plot_info["battery_name"] = battery_name.replace("_soc_data", "")

    result = {
        "battery_model": battery_name,
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "train_cycle_start": int(train_cycles[0]),
        "train_cycle_end": int(train_cycles[-1]),
        "val_cycle_start": int(val_cycles[0]),
        "val_cycle_end": int(val_cycles[-1]),
        "test_cycle_start": int(test_cycles[0]),
        "test_cycle_end": int(test_cycles[-1]),
        "num_train_cycles": int(len(train_cycles)),
        "num_val_cycles": int(len(val_cycles)),
        "num_test_cycles": int(len(test_cycles)),
        "best_val_mse": float(min(history["val_mse"])) if len(history["val_mse"]) > 0 else np.nan,
        "val_rmse": float(val_rmse),
        "val_mae": float(val_mae),
        "val_r2": float(val_r2),
        "val_physics_loss": float(val_phys_loss),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "test_inference_time_sec": float(test_inference_time_sec),
        "test_physics_loss": float(test_phys_loss),
        "epochs_ran": int(len(history["val_mse"])),
        "lambda_phys": float(LAMBDA_PHYS),
        "hidden_dim": int(HIDDEN_DIM),
        "learning_rate": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
        "seed": int(SEED),
    }

    print(
        f"[RESULT] {battery_name} | "
        f"test_rmse={test_rmse:.6f}, test_mae={test_mae:.6f}, "
        f"test_r2={test_r2:.6f}, "
        f"test_inference_time_sec={test_inference_time_sec:.6f}, "
        f"test_phys_loss={test_phys_loss:.6f}"
    )

    return result, plot_info


# ============================================================
# 12. Main
# ============================================================
def main() -> None:
    set_seed(SEED)

    all_results: List[Dict[str, float]] = []
    all_plot_infos: List[Dict[str, np.ndarray]] = []

    for csv_path in CSV_FILES:
        result, plot_info = run_one_battery(csv_path)
        all_results.append(result)
        all_plot_infos.append(plot_info)

    summary_df = pd.DataFrame(all_results)

    avg_row = {
        "battery_model": "AVERAGE",
        "total_samples": summary_df["total_samples"].mean(),
        "train_samples": summary_df["train_samples"].mean(),
        "val_samples": summary_df["val_samples"].mean(),
        "test_samples": summary_df["test_samples"].mean(),
        "train_cycle_start": np.nan,
        "train_cycle_end": np.nan,
        "val_cycle_start": np.nan,
        "val_cycle_end": np.nan,
        "test_cycle_start": np.nan,
        "test_cycle_end": np.nan,
        "num_train_cycles": summary_df["num_train_cycles"].mean(),
        "num_val_cycles": summary_df["num_val_cycles"].mean(),
        "num_test_cycles": summary_df["num_test_cycles"].mean(),
        "best_val_mse": summary_df["best_val_mse"].mean(),
        "val_rmse": summary_df["val_rmse"].mean(),
        "val_mae": summary_df["val_mae"].mean(),
        "val_r2": summary_df["val_r2"].mean(),
        "val_physics_loss": summary_df["val_physics_loss"].mean(),
        "test_rmse": summary_df["test_rmse"].mean(),
        "test_mae": summary_df["test_mae"].mean(),
        "test_r2": summary_df["test_r2"].mean(),
        "test_inference_time_sec": summary_df["test_inference_time_sec"].mean(),
        "test_physics_loss": summary_df["test_physics_loss"].mean(),
        "epochs_ran": summary_df["epochs_ran"].mean(),
        "lambda_phys": LAMBDA_PHYS,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([avg_row])], ignore_index=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    metrics_only_df = summary_df[
        ["battery_model", "test_rmse", "test_mae", "test_r2", "test_inference_time_sec"]
    ].copy()
    metrics_only_df.to_csv(OUTPUT_METRICS_ONLY_CSV, index=False)

    save_first_test_cycle_figure(
        plot_infos=all_plot_infos,
        save_path=OUTPUT_FIGURE,
    )

    print("\n================ Final Summary ================")
    print(summary_df)
    print(metrics_only_df)
    print(f"\n[INFO] Saved summary metrics to: {OUTPUT_SUMMARY_CSV}")
    print(f"[INFO] Saved metrics-only csv to: {OUTPUT_METRICS_ONLY_CSV}")
    print(f"[INFO] Saved figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()