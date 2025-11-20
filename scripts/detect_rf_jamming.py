import pandas as pd
import datetime as dt
from pathlib import Path

# -------- CONFIGURATION --------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = DATASETS_DIR / "flight_log.csv"
OUTPUT_EVENTS = RESULTS_DIR / "rf_jamming_events.csv"
OUTPUT_LOG = RESULTS_DIR / "rf_jamming_log.txt"

THRESHOLD = -75.0  # dBm – RF jamming threshold
# --------------------------------


def main():
    # Read flight telemetry
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    # Detect RF jamming events: low signal strength
    jamming_events = df[df["signal_strength"] < THRESHOLD].copy()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save detected events to CSV
    jamming_events.to_csv(OUTPUT_EVENTS, index=False)

    # Write forensic log
    with open(OUTPUT_LOG, "w") as log:
        log.write("RF Jamming Detection Log\n")
        log.write("-----------------------------------\n")
        log.write(f"Execution Time (UTC): {dt.datetime.utcnow().isoformat()}Z\n")
        log.write(f"Input File: {INPUT_FILE}\n")
        log.write(f"Output Events File: {OUTPUT_EVENTS}\n")
        log.write(f"Threshold: {THRESHOLD} dBm\n")
        log.write(f"Total Records Analyzed: {len(df)}\n")
        log.write(f"Total RF Jamming Events: {len(jamming_events)}\n\n")

        log.write("First detected events (up to 5 rows):\n")
        log.write("-----------------------------------\n")
        if len(jamming_events) > 0:
            log.write(jamming_events.head().to_string())
        else:
            log.write("No RF jamming events detected.\n")

    print("RF jamming detection completed successfully.")
    print(f"- Events saved to: {OUTPUT_EVENTS}")
    print(f"- Log saved to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
