
import pandas as pd
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

ML_RESULTS = RESULTS_DIR / "ml_predictions.csv"
FLIGHT_LOG = DATASETS_DIR / "flight_log.csv"
OUTPUT_FILE = RESULTS_DIR / "ml_telemetry_correlation.csv"

# Thresholds
GPS_THRESHOLD = 3.0
JAM_THRESHOLD = -75.0

def main():
    # Load telemetry
    flight = pd.read_csv(FLIGHT_LOG, parse_dates=["timestamp"])

    # Load ML results (must include a 'predicted_label' column)
    ml = pd.read_csv(ML_RESULTS, parse_dates=["timestamp"])

    # Ensure row counts match expected test set size
    if len(ml) != len(flight):
        print(" WARNING: ML result size does not match telemetry size.")
        print(len(ml), len(flight))

    # Combine ML results and telemetry
    df = pd.merge(flight, ml, on="timestamp", how="inner")

    # Forensic flags
    df["spoofing_flag"] = df["gps_drift"] > GPS_THRESHOLD
    df["jamming_flag"]  = df["signal_strength"] < JAM_THRESHOLD

    # Save combined correlation file
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"ML + Telemetry correlation exported to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
