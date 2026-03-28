from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd


# Paths
FEATURES_PATH = Path("cti_features.csv")
STIX_PATH = Path("OPSIGHT-main/CTI/stix/opsight_cti_stix_bundle.json")
OUTPUT_PATH = Path("cti_alerts.csv")

def evaluate_indicator(feature_row: dict[str, Any], indicator_name: str) -> tuple[bool, str]:
    name = indicator_name.lower()
    
    # --- Feature Extraction Mapping ---
    # Protocol Features
    inv_rate = float(feature_row.get("invalid_command_rate", 0) or 0)
    delta_sp = float(feature_row.get("avg_delta_setpoint", 0) or 0)
    pid_rate = float(feature_row.get("pid_modification_rate", 0) or 0)
    corr = float(feature_row.get("process_command_correlation", 0) or 0)
    freq = float(feature_row.get("command_frequency", 0) or 0)
    inter_std = float(feature_row.get("inter_command_std_ms", 0) or 0)
    
    # Human/Interface Behavioral Features (Aligned with new extraction)
    react_time = float(feature_row.get("avg_reaction_time_ms", 0) or 0)
    dwell_time = float(feature_row.get("avg_screen_dwell_ms", 0) or 0)

    # 1) Abnormal Setpoint Change Magnitude (STIX ID: indicator--a8ac4789...)
    if "abnormal setpoint change magnitude" in name:
        matched = delta_sp > 20
        return matched, f"avg_delta_setpoint={delta_sp:.2f} > 20"

    # 2) Unsolicited Response Without Preceding Command (STIX ID: indicator--0575a518...)
    # This is the primary indicator for injection attacks (Session 2 had -0.016)
    # 2) Unsolicited Response Logic (Split by specific attack characteristics)
    if "unsolicited response" in name:
        # Base condition: Injection attacks lack correlation between command and process state
        if corr < 0.10:
            
            # --- Sub-Logic to separate the 7 patterns ---
            
            # A) BURST: Very fast, overlapping frames
            if "burst" in name and inter_std < 5:
                return True, f"Burst timing detected: std={inter_std:.2f}ms"
            
            # B) FAST-RATE: Faster than normal 500ms-2s polling but not a burst
            if "fast-rate" in name and (5 <= inter_std < 100):
                return True, f"Fast injection rate: std={inter_std:.2f}ms"
            
            # C) SLOW-RATE: Blends in with normal traffic frequency
            if "slow-rate" in name and inter_std >= 100:
                # Session 2 usually falls here
                return True, f"Stealthy slow-rate injection: corr={corr:.4f}"
            
            # D) SETPOINT-TARGETED: Only if the injection involved a setpoint change
            if "setpoint-targeted" in name and delta_sp != 0:
                return True, f"Injection targeting setpoint: delta={delta_sp:.2f}"

            # E) SINGLE-PACKET: Only 1 packet (Hard to detect with aggregate features, usually frequency is very low)
            if "single-packet" in name and freq < 0.1:
                return True, f"Precision single-packet injection"

            # F) WAVE-PATTERN / NEGATIVE: (Requires checking min/max or oscillation)
            # If you don't have those features yet, you can leave these as 'False' 
            # to avoid cluttering the CSV.
            
        return False, ""
    
    # 3) Anomalous Burst Command/Response Timing (STIX ID: indicator--609fd3de...)
    if "anomalous burst" in name:
        matched = inter_std < 10 and freq > 1.5
        return matched, f"inter_std={inter_std:.2f}, freq={freq:.2f}"

    # 4) Unauthorized PID Parameter Change (STIX ID: indicator--5eae5f71...)
    if "pid parameter change" in name:
        matched = pid_rate > 0
        return matched, f"pid_modification_rate={pid_rate:.6f}"

    # 5) Malformed Modbus Frame Flood (STIX ID: indicator--e3d745ed...)
    if "malformed modbus frame flood" in name:
        matched = inv_rate > 0.5 and freq > 5.0
        return matched, f"invalid_rate={inv_rate:.2f}, freq={freq:.2f}"

    # 6) HUMAN BEHAVIOR: Critical Delay in Operator Reaction (New alignment)
    if "critical delay" in name or "slow reaction" in name:
        matched = react_time > 60000  # 60 seconds
        return matched, f"avg_reaction_time={react_time/1000:.2f}s"

    # 7) HUMAN BEHAVIOR: Excessive Screen Dwell
    if "screen dwell" in name:
        matched = dwell_time > 45000  # 45 seconds
        return matched, f"avg_dwell_time={dwell_time/1000:.2f}s"

    return False, ""

def correlate():
    if not FEATURES_PATH.exists() or not STIX_PATH.exists():
        print(" Error: Missing input files.")
        return

    features_df = pd.read_csv(FEATURES_PATH)
    bundle = json.load(open(STIX_PATH, "r", encoding="utf-8"))

    # Indexing
    object_index = {obj["id"]: obj for obj in bundle["objects"]}
    indicators = [obj for obj in bundle["objects"] if obj.get("type") == "indicator"]
    
    # Map Indicators to Attacks via Relationships
    indicator_to_attack = {}
    for rel in [obj for obj in bundle["objects"] if obj.get("type") == "relationship"]:
        if rel.get("relationship_type") == "indicates":
            indicator_to_attack.setdefault(rel["source_ref"], []).append(rel["target_ref"])

    results = []

    for _, row in features_df.iterrows():
        row_dict = row.to_dict()
        session_id = row_dict.get("SessionID")
        found_any_match = False

        for ind in indicators:
            matched, reason = evaluate_indicator(row_dict, ind.get("name", ""))
            if matched:
                found_any_match = True
                linked_attacks = indicator_to_attack.get(ind["id"], [])
                
                # If the indicator is linked to MITRE patterns, log each
                if linked_attacks:
                    for attack_id in linked_attacks:
                        atk = object_index.get(attack_id, {})
                        ext_refs = atk.get("external_references", [])
                        mitre_ids = ", ".join([r.get("external_id", "") for r in ext_refs if r.get("external_id")])
                        
                        results.append({
                            "SessionID": session_id,
                            "Indicator": ind["name"],
                            "Reason": reason,
                            "Attack_Pattern": atk.get("name"),
                            "MITRE_ICS_ID": mitre_ids
                        })
                else:
                    results.append({
                        "SessionID": session_id,
                        "Indicator": ind["name"],
                        "Reason": reason,
                        "Attack_Pattern": "No Linked Pattern",
                        "MITRE_ICS_ID": "N/A"
                    })

        if not found_any_match:
            results.append({
                "SessionID": session_id,
                "Indicator": "Normal Behavior",
                "Reason": "Baseline logic met",
                "Attack_Pattern": "None",
                "MITRE_ICS_ID": "N/A"
            })

    # Output
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*50)
    print("      OPSIGHT CTI CORRELATION REPORT")
    print("="*50)
    # Focus on threats in the console output
    threats = out_df[out_df["Indicator"] != "Normal Behavior"]
    if threats.empty:
        print(" No threats detected.")
    else:
        print(threats.to_string(index=False))
    print(f"\nFull log saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    correlate()
