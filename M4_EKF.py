import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)


# =========================================================
# 1. Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = [
    os.path.join(BASE_DIR, "datasets", "B0005_soc_data.csv"),
    os.path.join(BASE_DIR, "datasets", "B0006_soc_data.csv"),
    os.path.join(BASE_DIR, "datasets", "B0007_soc_data.csv"),
    os.path.join(BASE_DIR, "datasets", "B0018_soc_data.csv"),
]

RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "M4_EKF")

SPLIT_METHOD = "cycle"  # "cycle" or "sequential"
TEST_SIZE = 0.2
VAL_SIZE = 0.2

PROCESS_NOISE = 1e-4
MEASUREMENT_NOISE = 1e-3
OCV_POLY_DEGREE = 1

DEFAULT_Q_NOM = 2.0
DEFAULT_R_INTERNAL = 0.01

SAVE_PLOTS = True
TIME_SERIES_SAMPLE_LEN = 1000


# =========================================================
# 2. Utility Functions
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_soc_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soc"] = pd.to_numeric(df["soc"], errors="coerce")
    df = df.dropna(subset=["soc"])

    if len(df) == 0:
        return df

    if df["soc"].max() > 1.5:
        df["soc"] = df["soc"] / 100.0

    df["soc"] = df["soc"].clip(0.0, 1.0)
    return df


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if np.issubdtype(df["time"].dtype, np.number):
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        return df

    t_dt = pd.to_datetime(df["time"], errors="coerce")
    if t_dt.notna().all():
        t0 = t_dt.iloc[0]
        df["time"] = (t_dt - t0).dt.total_seconds()
        return df

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["cycle_number", "time", "voltage_measured", "current", "soc", "capacity"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df = normalize_time_column(df)
    df = normalize_soc_column(df)

    numeric_cols = [
        "cycle_number", "time", "voltage_measured", "current",
        "temperature", "soc", "capacity", "current_load", "voltage_load"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["cycle_number", "time", "voltage_measured", "current", "soc"])
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    return df


def split_by_cycles(df: pd.DataFrame, test_size=0.2, val_size=0.2):
    df = df.copy()
    cycles = sorted(df["cycle_number"].dropna().unique().tolist())

    n_cycles = len(cycles)
    if n_cycles < 3:
        return split_sequential(df, test_size=test_size, val_size=val_size)

    n_test = max(1, int(round(n_cycles * test_size)))
    n_val = max(1, int(round(n_cycles * val_size)))
    n_train = n_cycles - n_test - n_val

    if n_train < 1:
        n_train = max(1, n_cycles - 2)
        n_val = 1
        n_test = n_cycles - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = max(1, n_cycles - 2)

    train_cycles = cycles[:n_train]
    val_cycles = cycles[n_train:n_train + n_val]
    test_cycles = cycles[n_train + n_val:]

    train_df = df[df["cycle_number"].isin(train_cycles)].copy()
    val_df = df[df["cycle_number"].isin(val_cycles)].copy()
    test_df = df[df["cycle_number"].isin(test_cycles)].copy()

    return train_df, val_df, test_df


def split_sequential(df: pd.DataFrame, test_size=0.2, val_size=0.2):
    df = df.copy()
    n = len(df)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    n_train = n - n_test - n_val

    if n_train < 1:
        raise ValueError("Not enough data to complete the train/val/test split.")

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


def prepare_data(df: pd.DataFrame, split_method="cycle", test_size=0.2, val_size=0.2):
    if split_method == "cycle":
        return split_by_cycles(df, test_size=test_size, val_size=val_size)
    return split_sequential(df, test_size=test_size, val_size=val_size)


def save_metrics_txt(metrics: dict, save_path: str, dataset_name: str, test_inference_time: float):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"EKF Results - {dataset_name}\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"Test_Inference_Time_s: {test_inference_time:.6f}\n")


def get_first_cycle_data(df: pd.DataFrame, y_true, y_pred):
    df_plot = df.copy().reset_index(drop=True)
    df_plot["soc_true"] = np.asarray(y_true)
    df_plot["soc_pred"] = np.asarray(y_pred)

    first_cycle = sorted(df_plot["cycle_number"].unique())[0]
    cycle_df = df_plot[df_plot["cycle_number"] == first_cycle].copy().reset_index(drop=True)

    return first_cycle, cycle_df


# =========================================================
# 3. Plotting Functions
# =========================================================
def plot_ekf_comparison(y_true, y_pred, save_path, title="EKF SOC Estimation"):
    plt.figure(figsize=(10, 5))
    x = np.arange(len(y_true))

    plt.plot(x, y_true, label="Actual SOC", linewidth=2)
    plt.plot(x, y_pred, label="Predicted SOC", linewidth=2)

    plt.xlabel("Sample Index")
    plt.ylabel("SOC")
    plt.title(title)
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_first_test_cycle_4subplots(plot_data_list, save_path):
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
        ax.set_title(f"{dataset_name} - First Test Cycle ({first_cycle - 1})")
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


# =========================================================
# 4. EKF Class
# =========================================================
class ExtendedKalmanFilter:
    def __init__(self, q_nom, process_noise=1e-5, measurement_noise=1e-3):
        self.Q_nom = q_nom
        self.Q = process_noise
        self.R = measurement_noise

        self.ocv_model = None
        self.poly_features = None
        self.ocv_poly_degree = 5

        self.R_internal = DEFAULT_R_INTERNAL
        self.P0 = 0.1

    def fit_ocv_model(self, df_train: pd.DataFrame):
        soc = df_train["soc"].values
        voltage = df_train["voltage_measured"].values
        current = df_train["current"].values

        low_current_mask = np.abs(current) < 0.1
        if np.sum(low_current_mask) > 50:
            high_current_mask = np.abs(current) > 0.5
            if np.sum(high_current_mask) > 20:
                soc_bins = np.linspace(0.1, 0.9, 10)
                r_estimates = []

                for i in range(len(soc_bins) - 1):
                    bin_mask = (soc >= soc_bins[i]) & (soc < soc_bins[i + 1])
                    if np.sum(bin_mask) > 10:
                        bin_curr = current[bin_mask]
                        bin_volt = voltage[bin_mask]
                        valid = np.abs(bin_curr) > 0.1
                        if np.sum(valid) > 5:
                            r_est = np.median(
                                np.abs(bin_volt[valid] - np.median(bin_volt)) /
                                np.maximum(np.abs(bin_curr[valid]), 1e-8)
                            )
                            if 0.001 < r_est < 0.1:
                                r_estimates.append(r_est)

                if len(r_estimates) > 0:
                    self.R_internal = float(np.median(r_estimates))

        ocv_estimate = voltage + np.abs(current) * self.R_internal

        self.poly_features = PolynomialFeatures(degree=self.ocv_poly_degree)
        soc_poly = self.poly_features.fit_transform(soc.reshape(-1, 1))

        self.ocv_model = LinearRegression(fit_intercept=False)
        self.ocv_model.fit(soc_poly, ocv_estimate)

    def predict_ocv(self, soc):
        soc = np.array(soc).reshape(-1, 1)
        soc_poly = self.poly_features.transform(soc)
        return self.ocv_model.predict(soc_poly)

    def predict_voltage(self, soc, current):
        ocv = self.predict_ocv(soc)
        current = np.array(current)
        voltage = ocv + current * self.R_internal
        return voltage

    def compute_h_jacobian(self, soc):
        soc = np.array(soc)
        if soc.ndim == 0:
            soc = np.array([soc])

        coef = self.ocv_model.coef_
        coef_poly = coef[1:] if len(coef) > 1 else np.array([])

        h_jacobian = np.zeros_like(soc, dtype=float)
        for i in range(len(coef_poly)):
            h_jacobian += (i + 1) * coef_poly[i] * (soc ** i)

        if h_jacobian.size == 1:
            return float(h_jacobian.item())
        return h_jacobian

    def estimate(self, df: pd.DataFrame, initial_soc=None):
        time = df["time"].values
        current = -df["current"].values
        voltage_measured = df["voltage_measured"].values

        dt_diff = np.diff(time) / 3600.0
        dt = np.zeros(len(time))
        dt[:-1] = dt_diff
        dt[-1] = dt_diff[-1] if len(dt_diff) > 0 else 0.0

        n = len(df)
        soc_estimates = np.zeros(n)
        soc_covariance = np.zeros(n)

        if initial_soc is None:
            initial_soc = float(df["soc"].iloc[0]) if "soc" in df.columns else 1.0

        soc_est = float(np.clip(initial_soc, 0.0, 1.0))
        P = self.P0

        for k in range(n):
            F = 1.0
            soc_pred = soc_est + (current[k] * dt[k]) / self.Q_nom
            soc_pred = np.clip(soc_pred, 0.0, 1.0)
            P_pred = F * P * F + self.Q

            voltage_pred = self.predict_voltage(soc_pred, current[k])
            innovation = float(voltage_measured[k] - voltage_pred)
            H = self.compute_h_jacobian(soc_pred)
            S = H * P_pred * H + self.R
            K = (P_pred * H) / S if S != 0 else 0.0

            soc_est = soc_pred + K * innovation
            soc_est = float(np.clip(soc_est, 0.0, 1.0))
            P = (1 - K * H) * P_pred

            soc_estimates[k] = soc_est
            soc_covariance[k] = P

        return soc_estimates, soc_covariance


# =========================================================
# 5. Evaluation Functions
# =========================================================
def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def evaluate_ekf(ekf: ExtendedKalmanFilter, df: pd.DataFrame, initial_soc=None):
    y_pred, _ = ekf.estimate(df, initial_soc=initial_soc)
    y_true = df["soc"].values
    metrics = evaluate_predictions(y_true, y_pred)
    return metrics, y_true, y_pred


# =========================================================
# 6. Single-File Processing
# =========================================================
def process_one_dataset(file_path: str):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print("\n" + "=" * 70)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)
    df = clean_dataset(df)

    if len(df) < 50:
        print(f"[WARNING] Dataset is too small, skipped: {dataset_name}")
        return None

    print(f"Samples: {len(df)}")
    print(f"Cycles: {df['cycle_number'].nunique()}")
    print(f"SOC range: {df['soc'].min():.4f} ~ {df['soc'].max():.4f}")

    q_nom = pd.to_numeric(df["capacity"], errors="coerce").median()
    if pd.isna(q_nom) or q_nom <= 0:
        q_nom = DEFAULT_Q_NOM
    print(f"Nominal capacity Q_nom: {q_nom:.4f} Ah")

    train_df, val_df, test_df = prepare_data(
        df,
        split_method=SPLIT_METHOD,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    ekf = ExtendedKalmanFilter(
        q_nom=q_nom,
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE
    )
    ekf.ocv_poly_degree = OCV_POLY_DEGREE

    print("Fitting OCV-SOC model...")
    ekf.fit_ocv_model(train_df)
    print(f"Estimated internal resistance: {ekf.R_internal:.6f} Ohm")

    val_metrics, _, _ = evaluate_ekf(
        ekf,
        val_df,
        initial_soc=float(val_df["soc"].iloc[0])
    )
    print("Validation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.6f}")

    test_start_time = time.perf_counter()

    test_metrics, y_true, y_pred = evaluate_ekf(
        ekf,
        test_df,
        initial_soc=float(test_df["soc"].iloc[0])
    )

    test_end_time = time.perf_counter()
    test_inference_time = test_end_time - test_start_time

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"  Test Inference Time (s): {test_inference_time:.6f}")

    save_dir = os.path.join(RESULTS_DIR, dataset_name)
    ensure_dir(save_dir)

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["soc_pred_ekf"] = y_pred
    pred_df.to_csv(os.path.join(save_dir, "test_predictions.csv"), index=False)

    metrics_with_time = dict(test_metrics)
    metrics_with_time["Test_Inference_Time_s"] = test_inference_time

    metrics_df = pd.DataFrame([metrics_with_time])
    metrics_df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    save_metrics_txt(
        test_metrics,
        os.path.join(save_dir, "metrics.txt"),
        dataset_name,
        test_inference_time
    )

    # Restore the original behavior of saving one plot for each battery dataset.
    if SAVE_PLOTS:
        plot_ekf_comparison(
            y_true=y_true,
            y_pred=y_pred,
            save_path=os.path.join(save_dir, "ekf_comparison.png"),
            title=f"{dataset_name} - EKF SOC Estimation"
        )

    first_cycle, cycle_df = get_first_cycle_data(test_df, y_true, y_pred)

    return {
        "dataset": dataset_name,
        "RMSE": test_metrics["RMSE"],
        "MAE": test_metrics["MAE"],
        "R2": test_metrics["R2"],
        "Test_Inference_Time_s": test_inference_time,
        "plot_data": {
            "dataset_name": dataset_name.replace("_soc_data", ""),
            "first_cycle": first_cycle,
            "cycle_df": cycle_df
        }
    }


# =========================================================
# 7. Main Function
# =========================================================
def main():
    ensure_dir(RESULTS_DIR)

    all_results = []
    plot_data_list = []

    for file_path in DATA_FILES:
        result = process_one_dataset(file_path)
        if result is not None:
            all_results.append({
                "dataset": result["dataset"],
                "RMSE": result["RMSE"],
                "MAE": result["MAE"],
                "R2": result["R2"],
                "Test_Inference_Time_s": result["Test_Inference_Time_s"],
            })
            plot_data_list.append(result["plot_data"])

    if len(all_results) == 0:
        print("\nNo valid dataset processed.")
        return

    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    if SAVE_PLOTS and len(plot_data_list) > 0:
        plot_first_test_cycle_4subplots(
            plot_data_list=plot_data_list,
            save_path=os.path.join(RESULTS_DIR, "first_test_cycle_4subplots.png")
        )

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    if SAVE_PLOTS and len(plot_data_list) > 0:
        print(f"Saved plot to: {os.path.join(RESULTS_DIR, 'first_test_cycle_4subplots.png')}")


if __name__ == "__main__":
    main()