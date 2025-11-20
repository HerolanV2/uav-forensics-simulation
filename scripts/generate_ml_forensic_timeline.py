
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

INPUT_FILE = RESULTS_DIR / "ml_telemetry_correlation.csv"
OUTPUT_PLOT = RESULTS_DIR / "ml_forensic_timeline.png"


def main():
    # Load ML + telemetry correlation file
    df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

    # Map predicted labels to vertical positions
    label_map = {
        "normal": 0,
        "gps_spoofing": 1,
        "rf_jamming": -1
    }
    df["pred_level"] = df["predicted_label"].map(label_map)

    # Create plot
    plt.figure(figsize=(14, 5))

    # ML prediction layers
    plt.scatter(
        df[df["predicted_label"] == "normal"]["timestamp"],
        df[df["predicted_label"] == "normal"]["pred_level"],
        s=8, label="ML Prediction: Normal"
    )
    plt.scatter(
        df[df["predicted_label"] == "gps_spoofing"]["timestamp"],
        df[df["predicted_label"] == "gps_spoofing"]["pred_level"],
        s=12, color="red", label="ML Prediction: GPS Spoofing"
    )
    plt.scatter(
        df[df["predicted_label"] == "rf_jamming"]["timestamp"],
        df[df["predicted_label"] == "rf_jamming"]["pred_level"],
        s=12, color="blue", label="ML Prediction: RF Jamming"
    )

    # Forensic flags (telemetry based)
    spoof = df[df["spoofing_flag"] == True]
    jam = df[df["jamming_flag"] == True]

    if not spoof.empty:
        plt.scatter(
            spoof["timestamp"], [1.15] * len(spoof),
            marker="x", s=30, color="darkred",
            label="Forensic Flag: GPS Spoofing"
        )

    if not jam.empty:
        plt.scatter(
            jam["timestamp"], [-1.15] * len(jam),
            marker="x", s=30, color="navy",
            label="Forensic Flag: RF Jamming"
        )

    # Formatting
    plt.yticks(
        [-1.15, -1, 0, 1, 1.15],
        [
            "Forensic Jamming", "ML Jamming",
            "Normal",
            "ML Spoofing", "Forensic Spoofing"
        ]
    )
    plt.xlabel("Timestamp")
    plt.title("ML Prediction vs Telemetry Forensic Flags Timeline")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    # Save plot
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    print(f"Timeline saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
