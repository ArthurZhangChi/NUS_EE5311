import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# =========================================================
# Fixed input files
# =========================================================
INPUT_FILES = [
    "B0005_soc_data.csv",
    "B0006_soc_data.csv",
    "B0007_soc_data.csv",
    "B0018_soc_data.csv",
]

OUTPUT_DIR = "thevenin_all_cycles_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIT_STRIDE = 8  # subsampling for voltage parameter fitting only

# =========================================================
# Special selected cycles for 2x2 SOC plot
# First three batteries -> cycle 136
# Last battery          -> cycle 107
# =========================================================
SELECTED_CYCLES = {
    "B0005": 135,
    "B0006": 135,
    "B0007": 135,
    "B0018": 106,
}


# =========================================================
# Metrics
# =========================================================
def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# =========================================================
# OCV-SOC polynomial fit
# =========================================================
def fit_ocv_poly(soc, voltage, degree=7):
    mask = np.isfinite(soc) & np.isfinite(voltage)
    soc = np.asarray(soc[mask], dtype=float)
    voltage = np.asarray(voltage[mask], dtype=float)

    mask2 = (soc >= 0) & (soc <= 1) & (voltage >= 2.5) & (voltage <= 4.3)
    soc = soc[mask2]
    voltage = voltage[mask2]

    bins = np.linspace(0, 1, 101)
    idx = np.digitize(soc, bins) - 1

    xs, ys = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 3:
            xs.append(np.mean(soc[m]))
            ys.append(np.mean(voltage[m]))

    xs = np.array(xs)
    ys = np.array(ys)

    if len(xs) < 2:
        # fallback
        coef = np.polyfit([0, 1], [3.0, 4.2], deg=1)
        return np.poly1d(coef), pd.DataFrame({"soc_bin": [0, 1], "ocv_bin": [3.0, 4.2]})

    deg = min(degree, max(1, len(xs) - 1))
    coef = np.polyfit(xs, ys, deg=deg)
    return np.poly1d(coef), pd.DataFrame({"soc_bin": xs, "ocv_bin": ys})


# =========================================================
# SOC simulation by Coulomb counting
# Dataset sign convention:
# discharge current is negative
# SOC[k+1] = SOC[k] + I[k] * dt / Qn
# =========================================================
def simulate_soc_per_cycle(d):
    d = d.sort_values("time").reset_index(drop=True)

    t = d["time"].to_numpy(dtype=float)
    i = d["current"].to_numpy(dtype=float)
    soc_true = d["soc"].to_numpy(dtype=float)

    qn_as = max(float(2) * 3600.0, 1e-9)

    soc_est = np.zeros(len(d), dtype=float)
    soc_est[0] = soc_true[0]

    for k in range(len(d) - 1):
        dt = max(t[k + 1] - t[k], 0.0)
        soc_est[k + 1] = soc_est[k] + i[k] * dt / qn_as

    return np.clip(soc_est, 0.0, 1.0)


# =========================================================
# One-RC Thevenin voltage simulation
# Here OCV is evaluated using reference SOC column "soc"
# =========================================================
def simulate_voltage_grouped(df, ocv_func, params):
    R0, R1, C1 = params
    parts = []

    for cyc, d in df.groupby("cycle_number", sort=True):
        d = d.sort_values("time").copy()

        t = d["time"].to_numpy(dtype=float)
        i = d["current"].to_numpy(dtype=float)
        soc = d["soc"].to_numpy(dtype=float)

        vp = np.zeros(len(d), dtype=float)
        vpred = np.zeros(len(d), dtype=float)

        vpred[0] = ocv_func(soc[0]) - i[0] * R0

        inv_tau = 1.0 / (R1 * C1)
        inv_C1 = 1.0 / C1

        for k in range(len(d) - 1):
            dt = max(t[k + 1] - t[k], 0.0)
            vp[k + 1] = vp[k] + dt * (-inv_tau * vp[k] + inv_C1 * i[k])
            vpred[k + 1] = ocv_func(soc[k + 1]) - vp[k + 1] - i[k + 1] * R0

        d["vp_est"] = vp
        d["voltage_pred"] = vpred
        parts.append(d)

    return pd.concat(parts, ignore_index=True)


def voltage_residuals(params, fit_df, ocv_func):
    if np.min(params) <= 0:
        return np.full(len(fit_df), 1e3)

    pred = simulate_voltage_grouped(fit_df, ocv_func, params)["voltage_pred"].to_numpy(dtype=float)
    meas = fit_df.sort_values(["cycle_number", "time"])["voltage_measured"].to_numpy(dtype=float)
    return pred - meas


# =========================================================
# Analyze one battery file
# =========================================================
def analyze_battery(path):
    battery_name = os.path.basename(path).replace("_soc_data.csv", "")
    out_dir = os.path.join(OUTPUT_DIR, battery_name)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(path)

    required_cols = [
        "battery_id", "cycle_number", "time", "voltage_measured",
        "current", "temperature", "soc", "capacity"
    ]
    df = df[required_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(["cycle_number", "time"]).reset_index(drop=True)

    # -----------------------------------------------------
    # SOC analysis for all cycles
    # -----------------------------------------------------
    soc_t0 = time.perf_counter()
    
    soc_parts = []
    metrics = []

    for cyc, d in df.groupby("cycle_number", sort=True):
        d = d.sort_values("time").copy()
        d["soc_est"] = simulate_soc_per_cycle(d)
        d["soc_error"] = d["soc_est"] - d["soc"]

        soc_parts.append(d)

        metrics.append({
            "battery_id": battery_name,
            "cycle_number": int(cyc),
            "capacity_Ah": float(d["capacity"].iloc[0]),
            "soc_rmse": rmse(d["soc_est"], d["soc"]),
            "soc_mae": mae(d["soc_est"], d["soc"]),
            "soc_r2": r2_score_manual(d["soc"], d["soc_est"]),
            "soc_max_abs_error": float(np.max(np.abs(d["soc_error"]))),
        })

    soc_elapsed = time.perf_counter() - soc_t0
    print(f"{battery_name} SOC calculation time: {soc_elapsed:.6f} s")

    soc_df = pd.concat(soc_parts, ignore_index=True)
    cycle_summary = pd.DataFrame(metrics)

    # -----------------------------------------------------
    # OCV-SOC fit from quasi-rest points
    # -----------------------------------------------------
    rest = (df["current"].abs() < 0.02) & df["voltage_measured"].between(2.5, 4.3)
    ocv_func, ocv_bins = fit_ocv_poly(
        df.loc[rest, "soc"],
        df.loc[rest, "voltage_measured"],
        degree=7
    )
    ocv_bins.to_csv(os.path.join(out_dir, f"{battery_name}_ocv_bins.csv"), index=False)

    # -----------------------------------------------------
    # Fit Thevenin parameters using all cycles
    # subsampling only for optimization speed
    # -----------------------------------------------------
    fit_parts = []
    for cyc, d in df.groupby("cycle_number", sort=True):
        fit_parts.append(d.sort_values("time").iloc[::FIT_STRIDE].copy())
    fit_df = pd.concat(fit_parts, ignore_index=True)

    x0 = np.array([0.015, 0.01, 2000.0])   # [R0, R1, C1]
    lb = np.array([0.001, 0.001, 50.0])
    ub = np.array([0.1,   0.1,   20000.0])

    res = least_squares(
        voltage_residuals,
        x0,
        bounds=(lb, ub),
        args=(fit_df, ocv_func),
        max_nfev=25
    )

    params = res.x  # [R0, R1, C1]

    # -----------------------------------------------------
    # Full prediction on all points
    # -----------------------------------------------------
    full_pred = simulate_voltage_grouped(df, ocv_func, params)
    full_pred["voltage_error"] = full_pred["voltage_pred"] - full_pred["voltage_measured"]

    full_pred = full_pred.merge(
        soc_df[["cycle_number", "time", "soc_est", "soc_error"]],
        on=["cycle_number", "time"],
        how="left"
    )

    # -----------------------------------------------------
    # Voltage metrics per cycle
    # -----------------------------------------------------
    vmetrics = []
    for cyc, d in full_pred.groupby("cycle_number", sort=True):
        vmetrics.append({
            "battery_id": battery_name,
            "cycle_number": int(cyc),
            "voltage_rmse_V": rmse(d["voltage_pred"], d["voltage_measured"]),
            "voltage_mae_V": mae(d["voltage_pred"], d["voltage_measured"]),
            "voltage_r2": r2_score_manual(d["voltage_measured"], d["voltage_pred"]),
            "voltage_max_abs_error_V": float(np.max(np.abs(d["voltage_error"]))),
        })

    vmetrics = pd.DataFrame(vmetrics)
    cycle_summary = cycle_summary.merge(vmetrics, on=["battery_id", "cycle_number"], how="left")

    cycle_summary.to_csv(os.path.join(out_dir, f"{battery_name}_cycle_summary.csv"), index=False)
    full_pred.to_csv(os.path.join(out_dir, f"{battery_name}_all_cycles_predictions.csv"), index=False)

    # -----------------------------------------------------
    # Battery summary
    # -----------------------------------------------------
    cap = df.groupby("cycle_number")["capacity"].first().reset_index()

    batt_summary = {
        "battery_id": battery_name,
        "n_cycles": int(df["cycle_number"].nunique()),
        "R0_ohm": float(params[0]),
        "R1_ohm": float(params[1]),
        "C1_F": float(params[2]),
        "tau_s": float(params[1] * params[2]),
        "soc_rmse_mean": float(cycle_summary["soc_rmse"].mean()),
        "soc_mae_mean": float(cycle_summary["soc_mae"].mean()),
        "soc_r2_mean": float(cycle_summary["soc_r2"].mean()),
        "voltage_rmse_mean_V": float(cycle_summary["voltage_rmse_V"].mean()),
        "voltage_mae_mean_V": float(cycle_summary["voltage_mae_V"].mean()),
        "voltage_r2_mean": float(cycle_summary["voltage_r2"].mean()),
        "capacity_start_Ah": float(cap["capacity"].iloc[0]),
        "capacity_end_Ah": float(cap["capacity"].iloc[-1]),
    }

    # -----------------------------------------------------
    # Summary plots
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(cycle_summary["cycle_number"], cycle_summary["soc_rmse"])
    axes[0, 0].set_title(f"{battery_name}: SOC RMSE over all cycles")
    axes[0, 0].set_xlabel("Cycle number")
    axes[0, 0].set_ylabel("SOC RMSE")
    axes[0, 0].grid(True)

    axes[0, 1].plot(cycle_summary["cycle_number"], cycle_summary["voltage_rmse_V"])
    axes[0, 1].set_title(f"{battery_name}: Voltage RMSE over all cycles")
    axes[0, 1].set_xlabel("Cycle number")
    axes[0, 1].set_ylabel("Voltage RMSE (V)")
    axes[0, 1].grid(True)

    sg = np.linspace(0, 1, 200)
    axes[1, 0].scatter(ocv_bins["soc_bin"], ocv_bins["ocv_bin"], s=10)
    axes[1, 0].plot(sg, ocv_func(sg))
    axes[1, 0].set_title(f"{battery_name}: OCV-SOC fit")
    axes[1, 0].set_xlabel("SOC")
    axes[1, 0].set_ylabel("OCV (V)")
    axes[1, 0].grid(True)

    axes[1, 1].plot(cap["cycle_number"], cap["capacity"])
    axes[1, 1].set_title(f"{battery_name}: Capacity fade")
    axes[1, 1].set_xlabel("Cycle number")
    axes[1, 1].set_ylabel("Capacity (Ah)")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{battery_name}_summary_plots.png"), dpi=150)
    plt.close(fig)

    # -----------------------------------------------------
    # First / middle / last cycle plots
    # -----------------------------------------------------
    cyc_list = sorted(df["cycle_number"].unique())
    selected = [cyc_list[0], cyc_list[len(cyc_list) // 2], cyc_list[-1]]
    selected = list(dict.fromkeys(selected))

    fig, axes = plt.subplots(len(selected), 3, figsize=(16, 4 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    for r, cyc in enumerate(selected):
        d = full_pred[full_pred["cycle_number"] == cyc]

        axes[r, 0].plot(d["time"], d["current"])
        axes[r, 0].set_title(f"Cycle {cyc}: Current")
        axes[r, 0].grid(True)

        axes[r, 1].plot(d["time"], d["soc"], label="SOC true")
        axes[r, 1].plot(d["time"], d["soc_est"], "--", label="SOC est")
        axes[r, 1].set_title(f"Cycle {cyc}: SOC")
        axes[r, 1].grid(True)
        axes[r, 1].legend()

        axes[r, 2].plot(d["time"], d["voltage_measured"], label="Measured")
        axes[r, 2].plot(d["time"], d["voltage_pred"], "--", label="Predicted")
        axes[r, 2].set_title(f"Cycle {cyc}: Voltage")
        axes[r, 2].grid(True)
        axes[r, 2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{battery_name}_first_mid_last_cycles.png"), dpi=150)
    plt.close(fig)

    return batt_summary


# =========================================================
# Plot selected SOC cycles in one 2x2 figure
# B0005/B0006/B0007 -> cycle 136
# B0018              -> cycle 107
# =========================================================
def plot_selected_cycles_soc():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    for ax, path in zip(axes, INPUT_FILES):
        battery_name = os.path.basename(path).replace("_soc_data.csv", "")
        cycle_num = SELECTED_CYCLES[battery_name]

        df = pd.read_csv(path)
        d = (
            df[df["cycle_number"] == cycle_num]
            .sort_values("time")
            .reset_index(drop=True)
            .copy()
        )

        if d.empty:
            ax.set_title(f"{battery_name} - cycle {cycle_num} not found")
            ax.axis("off")
            continue

        soc_true = d["soc"].to_numpy(dtype=float)
        soc_pred = simulate_soc_per_cycle(d)

        ax.plot(np.arange(len(d)), soc_true, label="Actual SOC", linewidth=2)
        ax.plot(np.arange(len(d)), soc_pred, label="Predicted SOC", linewidth=2)

        ax.set_title(f"{battery_name} - First Test Cycle ({cycle_num})")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("SOC")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "selected_cycles_soc_plot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    rows = []

    for path in INPUT_FILES:
        print("Processing:", os.path.basename(path))
        rows.append(analyze_battery(path))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "all_battery_summary.csv"), index=False)

    print("\n===== All Battery Summary =====")
    print(summary)

    plot_selected_cycles_soc()
    print(f"\nDone. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()