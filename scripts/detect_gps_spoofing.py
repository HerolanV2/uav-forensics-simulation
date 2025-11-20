import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"
INPUT_FILE = DATASETS_DIR / "flight_log.csv"
OUTPUT_EVENTS = RESULTS_DIR / "gps_spoofing_events.csv"
OUTPUT_LOG = RESULTS_DIR / "gps_spoofing_log.txt"
OUTPUT_PLOT = RESULTS_DIR / "gps_drift_plot.png"
THRESHOLD = 3.0   

def main():
    # Load telemetry file
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    # Detect spoofing events
    spoof_events = df[df["gps_drift"] > THRESHOLD].copy()

    # Save extracted events
    spoof_events.to_csv(OUTPUT_EVENTS, index=False)

    # Create log file
    with open(OUTPUT_LOG, "w") as log:
        log.write("GPS Spoofing Detection Log\n")
        log.write("-----------------------------------\n")
        log.write(f"Execution Time (UTC): {datetime.utcnow()}\n")
        log.write(f"Input File: {INPUT_FILE}\n")
        log.write(f"Output File: {OUTPUT_EVENTS}\n")
        log.write(f"Threshold: {THRESHOLD} meters\n")
        log.write(f"Total Records Analyzed: {len(df)}\n")
        log.write(f"Total Spoofing Events: {len(spoof_events)}\n\n")

        # Write first few spoof events for preview
        log.write("First 5 detected events:\n")
        log.write("-----------------------------------\n")
        if len(spoof_events) > 0:
            log.write(spoof_events.head().to_string())
        else:
            log.write("No spoofing events detected.\n")

    print("GPS spoofing detection completed.")
    print(f"Events saved to: {OUTPUT_EVENTS}")
    print(f"Log saved to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()


plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["gps_drift"], label="GPS Drift")
plt.axhline(3.0, color="red", linestyle="--", label="Spoofing Threshold")
plt.xlabel("Time")
plt.ylabel("GPS Drift (meters)")
plt.title("GPS Drift Over Time (Spoofing Detection)")
plt.legend()
plt.tight_layout()
plt.show()

