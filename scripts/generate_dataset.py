#!/usr/bin/env python3
"""
generate_dataset.py
Simulated drone flight dataset generator for Practicum:
labels: normal, gps_spoofing, rf_jamming
Outputs: drone_simulation_dataset.csv and generation log
"""

import os
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
RANDOM_SEED = 42
N_PER_CLASS = 500           # total rows = 3 * N_PER_CLASS (changeable)
OUT_CSV = "datasets/drone_simulation_dataset.csv"
LOG_FILE = "results/generation_output.txt"
# ----------------------------

np.random.seed(RANDOM_SEED)

def make_normal(n):
    lat = np.cumsum(np.random.normal(0.00003, 0.00015, n)) + 53.3498
    lon = np.cumsum(np.random.normal(0.00002, 0.00015, n)) - 6.2603
    altitude = np.random.normal(120.0, 3.0, n)
    speed = np.abs(np.random.normal(8.0, 0.8, n))
    heading = np.mod(np.cumsum(np.random.normal(0.5, 1.0, n)), 360)
    signal_strength = np.random.normal(-60, 2.0, n)   # dBm
    gps_drift = np.abs(np.random.normal(0.5, 0.2, n))  # meters
    return pd.DataFrame({
        "latitude": lat, "longitude": lon, "altitude": altitude,
        "speed": speed, "heading": heading,
        "signal_strength": signal_strength, "gps_drift": gps_drift,
        "label": ["normal"] * n
    })

def make_gps_spoofing(n):
    # sudden jumps in lat/lon (large drift), inconsistent heading
    lat = np.cumsum(np.random.normal(0.0001, 0.0005, n)) + 53.3498
    lon = np.cumsum(np.random.normal(0.0001, 0.0005, n)) - 6.2603
    altitude = np.random.normal(120.0, 6.0, n)
    speed = np.abs(np.random.normal(9.0, 2.0, n))  # sometimes higher during spoof
    heading = np.mod(np.cumsum(np.random.normal(5.0, 10.0, n)), 360)
    signal_strength = np.random.normal(-62, 3.0, n)
    gps_drift = np.abs(np.random.normal(8.0, 3.0, n))  # much larger drift
    return pd.DataFrame({
        "latitude": lat, "longitude": lon, "altitude": altitude,
        "speed": speed, "heading": heading,
        "signal_strength": signal_strength, "gps_drift": gps_drift,
        "label": ["gps_spoofing"] * n
    })

def make_rf_jamming(n):
    # signal drops, sudden speed decrease/altitude wobble
    lat = np.cumsum(np.random.normal(0.00004, 0.0002, n)) + 53.3498
    lon = np.cumsum(np.random.normal(0.00003, 0.0002, n)) - 6.2603
    altitude = np.random.normal(120.0, 8.0, n) + np.random.normal(0, 3.0, n)
    speed = np.abs(np.random.normal(6.0, 2.5, n))  # slower on average
    heading = np.mod(np.cumsum(np.random.normal(1.0, 6.0, n)), 360)
    # intermittent deep drops in signal strength
    base = np.random.normal(-70, 4.0, n)
    bursts = (np.random.rand(n) < 0.15).astype(int) * np.random.normal(-20, 5.0, n)
    signal_strength = base + bursts
    gps_drift = np.abs(np.random.normal(2.0, 1.5, n))
    return pd.DataFrame({
        "latitude": lat, "longitude": lon, "altitude": altitude,
        "speed": speed, "heading": heading,
        "signal_strength": signal_strength, "gps_drift": gps_drift,
        "label": ["rf_jamming"] * n
    })

def main():
    start = datetime.utcnow()
    with open(LOG_FILE, "w") as log:
        log.write(f"Generation start (UTC): {start.isoformat()}\n")
        log.write(f"RANDOM_SEED={RANDOM_SEED}\n")
        log.write(f"N_PER_CLASS={N_PER_CLASS}\n\n")
        # generate
        df_normal = make_normal(N_PER_CLASS)
        df_spoof = make_gps_spoofing(N_PER_CLASS)
        df_jam = make_rf_jamming(N_PER_CLASS)
        df = pd.concat([df_normal, df_spoof, df_jam], ignore_index=True)
        # shuffle
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        end = datetime.utcnow()
        log.write(f"Saved {OUT_CSV} with {len(df)} rows\n")
        log.write(f"Generation end (UTC): {end.isoformat()}\n")
        # compute sha256
        sha256 = hashlib.sha256(open(OUT_CSV,"rb").read()).hexdigest()
        log.write(f"SHA256: {sha256}\n")
    print("Generation complete.")
    print(f"CSV: {OUT_CSV}, rows: {len(df)}")
    print(f"Log: {LOG_FILE} created.")
    print(f"SHA256: {sha256}")

if __name__ == "__main__":
    main()

