
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of telemetry records to generate
N = 1500  

# Start time of the simulated flight (UTC)
start_time = datetime(2025, 1, 1, 12, 0, 0)

# Create timestamp list (1-second intervals)
timestamps = [start_time + timedelta(seconds=i) for i in range(N)]

# Generate simulated telemetry values
df = pd.DataFrame({
    "timestamp": timestamps,
    "latitude": np.random.normal(53.3498, 0.0005, N),
    "longitude": np.random.normal(-6.2603, 0.0005, N),
    "altitude": np.random.normal(120, 3, N),
    "speed": np.abs(np.random.normal(8, 1, N)),
    "heading": np.random.uniform(0, 360, N),
    "signal_strength": np.random.normal(-60, 5, N),
    "gps_drift": np.abs(np.random.normal(1.5, 1.0, N))
})

# Save generated telemetry to CSV
df.to_csv("flight_log.csv", index=False)
print("flight_log.csv has been successfully generated.")
