import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

# Total number of telemetry records (1 Hz sampling → 1500 sec ≈ 25 minutes)
N = 1500

N_NORMAL = 500
N_SPOOFING = 500
N_JAMMING = N - N_NORMAL - N_SPOOFING  # 500

# Start time of the simulated flight (UTC)
start_time = datetime(2025, 1, 1, 12, 0, 0)


def main():
    # --- 1. Create timestamp series ---
    timestamps = [start_time + timedelta(seconds=i) for i in range(N)]

    # --- 2. Base telemetry (shared behaviour) ---
    # Latitude / longitude around Dublin (random small noise)
    latitude = np.random.normal(53.3498, 0.0005, N)
    longitude = np.random.normal(-6.2603, 0.0005, N)

    # Altitude and speed with small noise across whole flight
    altitude = np.random.normal(120, 3, N)
    speed = np.abs(np.random.normal(8, 1, N))
    heading = np.random.uniform(0, 360, N)

    # --- 3. GPS drift profile (Normal → Spoofing → Normal/Jamming) ---
    gps_drift = np.zeros(N)

    # Normal segment: low drift
    gps_drift[:N_NORMAL] = np.abs(np.random.normal(1.0, 0.4, N_NORMAL))

    # Spoofing segment: high drift (above threshold ≈ 3m)
    gps_drift[N_NORMAL:N_NORMAL + N_SPOOFING] = np.abs(
        np.random.normal(4.0, 0.7, N_SPOOFING)
    )

    # Jamming segment: drift returns closer to normal (spoofing yok)
    gps_drift[N_NORMAL + N_SPOOFING:] = np.abs(
        np.random.normal(1.2, 0.5, N_JAMMING)
    )

    # --- 4. Signal strength profile (Normal → Normal → Jamming) ---
    signal_strength = np.zeros(N)

    # Normal + spoofing segment: typical RSSI around -60 dBm
    signal_strength[:N_NORMAL + N_SPOOFING] = np.random.normal(-60, 3, N_NORMAL + N_SPOOFING)

    # Jamming segment: strong interference → very low signal strength (< -75 dBm)
    signal_strength[N_NORMAL + N_SPOOFING:] = np.random.normal(-82, 2, N_JAMMING)

    # --- 5. Build DataFrame ---
    df = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "speed": speed,
        "heading": heading,
        "signal_strength": signal_strength,
        "gps_drift": gps_drift,
    })

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASETS_DIR / "flight_log.csv", index=False)
    print("flight_log.csv has been successfully generated with normal, spoofing, and jamming segments.")


if __name__ == "__main__":
    main()
