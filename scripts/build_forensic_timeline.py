
import pandas as pd
import matplotlib.pyplot as plt

FLIGHT_FILE = "datasets/flight_log.csv"
SPOOF_FILE = "results/gps_spoofing_events.csv"
JAM_FILE = "results/rf_jamming_events.csv"

OUTPUT_TIMELINE_CSV = "results/forensic_timeline.csv"
OUTPUT_TIMELINE_PLOT = "results/forensic_timeline.png"


def main():
    # Load base telemetry (used only for full time range)
    flight = pd.read_csv(FLIGHT_FILE, parse_dates=["timestamp"])

    # Load detected events
    spoof = pd.read_csv(SPOOF_FILE, parse_dates=["timestamp"])
    jam = pd.read_csv(JAM_FILE, parse_dates=["timestamp"])

    # Create a simple event-type dataframe for spoofing
    spoof_timeline = pd.DataFrame({
        "timestamp": spoof["timestamp"],
        "event_type": "gps_spoofing"
    })

    # Create a simple event-type dataframe for jamming
    jam_timeline = pd.DataFrame({
        "timestamp": jam["timestamp"],
        "event_type": "rf_jamming"
    })

    # Concatenate into a single timeline
    timeline = pd.concat([spoof_timeline, jam_timeline], ignore_index=True)
    timeline = timeline.sort_values("timestamp").reset_index(drop=True)

    # Save as CSV
    timeline.to_csv(OUTPUT_TIMELINE_CSV, index=False)

    # ---- Build visual timeline plot ----
    plt.figure(figsize=(14, 4))

    # Draw a neutral baseline for the full flight duration
    plt.plot(flight["timestamp"], [0] * len(flight),
             color="lightgray", linewidth=1)

    # Map event types to y positions
    # +1 = spoofing, -1 = jamming
    if not spoof.empty:
        plt.scatter(
            spoof["timestamp"],
            [1] * len(spoof),
            color="red",
            s=12,
            label="GPS Spoofing"
        )

    if not jam.empty:
        plt.scatter(
            jam["timestamp"],
            [-1] * len(jam),
            color="blue",
            s=12,
            label="RF Jamming"
        )

    plt.yticks([-1, 0, 1], ["Jamming", "Normal", "Spoofing"])
    plt.xlabel("Time")
    plt.title("Combined Forensic Timeline (Spoofing + Jamming Events)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_TIMELINE_PLOT, dpi=200)
    plt.close()

    print("Forensic timeline generated successfully.")
    print(f"- Timeline CSV:  {OUTPUT_TIMELINE_CSV}")
    print(f"- Timeline Plot: {OUTPUT_TIMELINE_PLOT}")


if __name__ == "__main__":
    main()

