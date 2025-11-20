
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


# -------- CONFIGURATION --------
INPUT_FILE = "flight_log.csv"
OUTPUT_EVENTS = "gps_spoofing_events.csv"
OUTPUT_LOG = "gps_spoofing_log.txt"
OUTPUT_PLOT = "gps_drift_plot.png"
THRESHOLD = 3.0  # meters
# --------------------------------


def main():

    # Load telemetry CSV
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    # Detect spoofing events
    spoof_events = df[df["gps_drift"] > THRESHOLD].copy()

    # Save events to CSV
    spoof_events.to_csv(OUTPUT_EVENTS, index=False)

    # Generate and save plot
    plt.figure(figsize=(12, 4))
    plt.plot(df["timestamp"], df["gps_drift"], label="GPS Drift")
    plt.axhline(THRESHOLD, color="red", linestyle="--", label="Spoofing Threshold")
    plt.xlabel("Time")
    plt.ylabel("GPS Drift (meters)")
    plt.title("GPS Drift Over Time (Spoofing Detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()

    # Create forensic log
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log.write("GPS Spoofing Detection Log\n")
        log.write("-----------------------------------\n")
        log.write(f"Execution Time (UTC): {dt.datetime.utcnow().isoformat()}Z\n")
        log.write(f"Input File: {INPUT_FILE}\n")
        log.write(f"Output Events File: {OUTPUT_EVENTS}\n")
        log.write(f"Output Plot File: {OUTPUT_PLOT}\n")
        log.write(f"Threshold: {THRESHOLD} meters\n")
        log.write(f"Total Records Analyzed: {len(df)}\n")
        log.write(f"Total Spoofing Events: {len(spoof_events)}\n\n")

        # First 5 event preview
        log.write("First 5 spoofing events:\n")
        log.write("-----------------------------------\n")
        if len(spoof_events) > 0:
            log.write(spoof_events.head().to_string())
        else:
            log.write("No spoofing events detected.\n")

    print("GPS spoofing detection completed successfully.")
    print(f"- Events saved to: {OUTPUT_EVENTS}")
    print(f"- Log saved to: {OUTPUT_LOG}")
    print(f"- Plot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
