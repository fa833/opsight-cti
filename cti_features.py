
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd



INPUT_XLSX  = Path(r"C:\Users\alrom\Downloads\OP01_sessions.xlsx")
OUTPUT_CSV  = Path("cti_features.csv")

# =========================
# Helpers
# =========================
def normalize_fc(x: Any) -> int | None:
    """Normalize FunctionCode values like '0x3' or 3 to int."""
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if not s:
        return None
    try:
        return int(s, 16) if s.startswith("0x") else int(float(s))
    except (ValueError, TypeError):
        return None


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series: pd.Series) -> float:
    s = safe_num(series).dropna()
    return float(s.mean()) if not s.empty else 0.0


def safe_median(series: pd.Series) -> float:
    s = safe_num(series).dropna()
    return float(s.median()) if not s.empty else 0.0


def safe_std(series: pd.Series) -> float:
    s = safe_num(series).dropna()
    return float(s.std(ddof=0)) if not s.empty else 0.0


def entropy_from_series(series: pd.Series) -> float:
    vals = series.dropna().astype(str)
    if vals.empty:
        return 0.0
    probs = vals.value_counts(normalize=True)
    ent = float(-(probs * np.log2(probs)).sum())
    return max(0.0, ent)  # clamp floating-point artifact like -0.0


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def count_state_changes(series: pd.Series) -> int:
    s = series.fillna("").astype(str)
    if len(s) <= 1:
        return 0
    return max(int(s.ne(s.shift()).sum() - 1), 0)


def get_high_risk_mask(df: pd.DataFrame) -> pd.Series:
    """
    Explicit allowlist of risky operator actions.
    Excludes auto_poll by construction.#####
    """
    action_text = df["ActionType"].fillna("").astype(str).str.lower()

    allowed_prefixes = (
        "pressure_sp",
        "setpoint",
        "pump_",
        "valve",
        "mode_",
        "ack",
        "manual",
        "auto_mode",
        "shutdown",
        "startup",
    )

    action_based = action_text.apply(lambda x: x.startswith(allowed_prefixes))

    fc_write = df["FunctionCode_norm"].eq(0x10)

    pid_change_sum = (
        safe_num(df["deltaPIDGain"]).fillna(0).abs()
        + safe_num(df["deltaPIDReset"]).fillna(0).abs()
        + safe_num(df["deltaPIDRate"]).fillna(0).abs()
        + safe_num(df["deltaPIDDeadband"]).fillna(0).abs()
        + safe_num(df["deltaPIDCycleTime"]).fillna(0).abs()
    )
    pid_based = pid_change_sum > 0

    setpoint_based = safe_num(df["deltaSetPoint"]).fillna(0).abs() > 0

    return action_based | fc_write | pid_based | setpoint_based


def compute_process_corr(sp_abs: pd.Series, psi_abs: pd.Series) -> float:
    if len(sp_abs) >= 2 and sp_abs.nunique() > 1 and psi_abs.nunique() > 1:
        corr = float(sp_abs.corr(psi_abs))
        if math.isnan(corr):
            return 0.0
        return corr
    return 0.0


def compute_phase_scoped_corr(df: pd.DataFrame) -> float:
    if "Phase" not in df.columns:
        return 0.0

    phase_corrs = []
    for _, phase_df in df.groupby("Phase"):
        sp_abs = safe_num(phase_df["deltaSetPoint"]).fillna(0).abs()
        psi_abs = safe_num(phase_df["deltaPipelinePSI"]).fillna(0).abs()
        corr = compute_process_corr(sp_abs, psi_abs)
        if not math.isnan(corr):
            phase_corrs.append(corr)

    if not phase_corrs:
        return 0.0
    return float(np.mean(phase_corrs))


# =========================
# Main per-session extraction
# =========================
def extract_session_features(df: pd.DataFrame, sheet_name: str) -> dict[str, Any]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").reset_index(drop=True)

    df["FunctionCode_norm"] = df["FunctionCode"].apply(normalize_fc) if "FunctionCode" in df.columns else None
    df["is_write"] = df["FunctionCode_norm"].eq(0x10)
    df["is_read"] = df["FunctionCode_norm"].eq(0x03)

    invalid_data = ~df["InvalidDataLength"].fillna("X").astype(str).str.upper().eq("X") if "InvalidDataLength" in df.columns else False
    invalid_fc = ~df["InvalidFunctionCode"].fillna("X").astype(str).str.upper().eq("X") if "InvalidFunctionCode" in df.columns else False
    df["is_invalid"] = invalid_data | invalid_fc

    # Duration
    if "Timestamp" in df.columns and df["Timestamp"].notna().sum() >= 2:
        duration_sec = float((df["Timestamp"].max() - df["Timestamp"].min()).total_seconds())
    else:
        duration_sec = float(safe_num(df.get("TimeInterval", pd.Series(dtype=float))).fillna(0).sum() / 1000.0)

    duration_sec = max(duration_sec, 1.0)

    # PID changes
    pid_change_sum = (
        safe_num(df.get("deltaPIDGain", pd.Series(dtype=float))).fillna(0).abs()
        + safe_num(df.get("deltaPIDReset", pd.Series(dtype=float))).fillna(0).abs()
        + safe_num(df.get("deltaPIDRate", pd.Series(dtype=float))).fillna(0).abs()
        + safe_num(df.get("deltaPIDDeadband", pd.Series(dtype=float))).fillna(0).abs()
        + safe_num(df.get("deltaPIDCycleTime", pd.Series(dtype=float))).fillna(0).abs()
    )
    df["pid_modified"] = pid_change_sum > 0

    # Setpoint
    delta_sp = safe_num(df.get("deltaSetPoint", pd.Series(dtype=float))).fillna(0)
    df["setpoint_changed"] = delta_sp.abs() > 0
    df["setpoint_shock"] = delta_sp.abs() >= 15  # conservative lower bound

    # State changes
    mode_changes = count_state_changes(df.get("ControlMode", pd.Series(dtype=object)))
    pump_changes = count_state_changes(df.get("PumpState", pd.Series(dtype=object)))
    screen_changes = count_state_changes(df.get("ActiveScreen", pd.Series(dtype=object)))

    # High-risk actions
    high_risk_mask = get_high_risk_mask(df)
    total_actions = int(high_risk_mask.sum())

    alarm_viewed = safe_num(df.get("AlarmViewedBefore", pd.Series(dtype=float))).fillna(0)
    trend_viewed = safe_num(df.get("TrendViewedBefore", pd.Series(dtype=float))).fillna(0)

    informed_mask = (alarm_viewed == 1) | (trend_viewed == 1)
    no_context_mask = (alarm_viewed == 0) & (trend_viewed == 0)

    informed_high_risk_actions = int((high_risk_mask & informed_mask).sum())
    no_context_high_risk_actions = int((high_risk_mask & no_context_mask).sum())

    # Use auto_poll rows for interaction gap statistics
    action_text = df["ActionType"].fillna("").astype(str).str.lower()
    auto_poll_rows = df[action_text.eq("auto_poll")].copy()
    interaction_gap_auto = safe_num(auto_poll_rows.get("InteractionGap_ms", pd.Series(dtype=float))).dropna()
    interaction_gap_auto = interaction_gap_auto[interaction_gap_auto > 0]  # exclude startup zeros

    # Reactions
    reaction_times = safe_num(df.get("ReactionTime_ms", pd.Series(dtype=float))).dropna()
    reaction_times = reaction_times[reaction_times > 0]

    # Dwell
    screen_dwell = safe_num(df.get("ScreenDwellTime_ms", pd.Series(dtype=float))).dropna()
    clicks_before_action = safe_num(df.get("ClicksBeforeAction", pd.Series(dtype=float))).dropna()

    # Entropy
    command_entropy = entropy_from_series(df.get("ActionType", pd.Series(dtype=object)))
    function_code_entropy = entropy_from_series(df.get("FunctionCode", pd.Series(dtype=object)))

    # Ratios
    high_risk_command_ratio = safe_ratio(total_actions, len(df))
    invalid_command_rate = float(df["is_invalid"].mean()) if len(df) else 0.0
    write_ratio = float(df["is_write"].mean()) if len(df) else 0.0
    read_ratio = float(df["is_read"].mean()) if len(df) else 0.0

    setpoint_change_rate = safe_ratio(int(df["setpoint_changed"].sum()), duration_sec)
    setpoint_shock_event_rate = safe_ratio(int(df["setpoint_shock"].sum()), duration_sec)
    pid_modification_rate = safe_ratio(int(df["pid_modified"].sum()), duration_sec)
    control_mode_change_rate = safe_ratio(mode_changes, duration_sec)
    pump_state_change_rate = safe_ratio(pump_changes, duration_sec)
    screen_switch_rate = safe_ratio(screen_changes, duration_sec)

    trend_usage_ratio = float((trend_viewed == 1).mean()) if len(df) else 0.0
    alarm_usage_ratio = float((alarm_viewed == 1).mean()) if len(df) else 0.0
    informed_action_ratio = safe_ratio(informed_high_risk_actions, total_actions)
    no_context_action_ratio = safe_ratio(no_context_high_risk_actions, total_actions)

    avg_reaction_time_ms = safe_mean(reaction_times)
    fast_reaction_ratio = float((reaction_times < 500).mean()) if not reaction_times.empty else 0.0
    slow_reaction_ratio = float((reaction_times > 10000).mean()) if not reaction_times.empty else 0.0

    # Process-command correlation
    delta_psi = safe_num(df.get("deltaPipelinePSI", pd.Series(dtype=float))).fillna(0)
    process_command_correlation = compute_process_corr(delta_sp.abs(), delta_psi.abs())
    phase_scoped_process_corr = compute_phase_scoped_corr(df)

    operator_id = (
        df["OperatorID"].dropna().astype(str).iloc[0]
        if "OperatorID" in df.columns and not df["OperatorID"].dropna().empty
        else "UNKNOWN"
    )

    session_id = (
        df["SessionID"].dropna().iloc[0]
        if "SessionID" in df.columns and not df["SessionID"].dropna().empty
        else sheet_name
    )

    label_mode = (
        df["Label"].mode().iloc[0]
        if "Label" in df.columns and not df["Label"].mode().empty
        else ""
    )

    return {
        "sheet_name": sheet_name,
        "OperatorID": operator_id,
        "SessionID": session_id,
        "rows": int(len(df)),
        "duration_sec": round(duration_sec, 2),

        # protocol/process CTI features
        "command_frequency": round(len(df) / duration_sec, 6),
        "read_ratio": round(read_ratio, 6),
        "write_ratio": round(write_ratio, 6),
        "high_risk_command_ratio": round(high_risk_command_ratio, 6),
        "invalid_command_rate": round(invalid_command_rate, 6),
        "setpoint_change_rate": round(setpoint_change_rate, 6),
        "setpoint_shock_event_rate": round(setpoint_shock_event_rate, 6),
        "pid_modification_rate": round(pid_modification_rate, 6),
        "control_mode_change_rate": round(control_mode_change_rate, 6),
        "pump_state_change_rate": round(pump_state_change_rate, 6),
        "command_entropy": round(command_entropy, 6),
        "function_code_entropy": round(function_code_entropy, 6),
        "inter_command_mean_ms": round(safe_mean(interaction_gap_auto), 3),
        "inter_command_std_ms": round(safe_std(interaction_gap_auto), 3),
        "process_command_correlation": round(process_command_correlation, 6),
        "phase_scoped_process_corr": round(phase_scoped_process_corr, 6),
        "avg_delta_setpoint": round(safe_mean(delta_sp.abs()), 6),
        "avg_delta_pipeline_psi": round(safe_mean(delta_psi.abs()), 6),

        # HMI behavioral CTI features
        "avg_screen_dwell_ms": round(safe_mean(screen_dwell), 3),
        "median_screen_dwell_ms": round(safe_median(screen_dwell), 3),
        "pct_dwell_over_30s": round(float((screen_dwell > 30000).mean()) if not screen_dwell.empty else 0.0, 6),
        "screen_switch_rate": round(screen_switch_rate, 6),
        "avg_clicks_before_action": round(safe_mean(clicks_before_action), 3),
        "trend_usage_ratio": round(trend_usage_ratio, 6),
        "alarm_usage_ratio": round(alarm_usage_ratio, 6),
        "informed_action_ratio": round(informed_action_ratio, 6),
        "no_context_action_ratio": round(no_context_action_ratio, 6),
        "no_context_high_risk_actions": no_context_high_risk_actions,
        "avg_reaction_time_ms": round(avg_reaction_time_ms, 3),
        "fast_reaction_ratio": round(fast_reaction_ratio, 6),
        "slow_reaction_ratio": round(slow_reaction_ratio, 6),

        "label_mode": label_mode,
    }


# =========================
# Runner
# =========================
def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_XLSX}")

    xls = pd.ExcelFile(INPUT_XLSX)
    rows: list[dict[str, Any]] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(INPUT_XLSX, sheet_name=sheet)
        # Assuming extract_session_features is defined elsewhere
        rows.append(extract_session_features(df, sheet))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(" CTI FEATURE EXTRACTION COMPLETE")
    print("=" * 80)

    # List of columns you actually care about seeing in the terminal
    important_cols = [
        "sheet_name", "OperatorID", "SessionID", "rows", "duration_sec",
        "command_frequency", "write_ratio", "command_entropy", 
        "process_command_correlation", "informed_action_ratio", 
        "avg_reaction_time_ms", "slow_reaction_ratio", "label_mode"
    ]

    print("\n SUMMARY :")
    print("-" * 80)
    
    # We use .T (Transpose) to flip the table so it doesn't wrap
    # We only show the important_cols to keep it clean
    if not out.empty:
        summary_df = out[important_cols].copy()
        print(summary_df.T)
    else:
        print("No data processed.")

    print("-" * 80)
    print(f"\n FULL DATA SAVED TO: {OUTPUT_CSV.resolve()}")
    print("Tip: Open the CSV in Excel to see all features in a grid.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
