# UAV Forensics Dataset Generator

This repository includes a Python script for the generation of the simulated UAV flight dataset to be used for forensic analysis against GPS spoofing and RF jamming attacks, in a programmable way.

## Files
### Dataset Files (`/datasets`)
- `drone_simulation_dataset.csv`: Synthetic dataset used for machine learning classification.
- `flight_log.csv`: Simulated UAV telemetry dataset used for forensic analysis.

### Script Files (`/scripts`)
- `generate_dataset.py`: Script that generates the machine learning dataset.
- `generate_flight_log.py`: Script that generates the telemetry (flight log) dataset.
- `classification_model.py`: Machine learning model training and evaluation script.
- `generate_forensic_timeline.py`: Script that generates the combined forensic timeline (spoofing + jamming).

### Result Files (`/results`)
- `classification_results.csv`: Performance metrics (accuracy, precision, recall, F1-score).
- `Random_Forest_confusion_matrix.png`: Confusion matrix for Random Forest model.
- `SVM_confusion_matrix.png`: Confusion matrix for SVM model.
- `Logistic_Regression_confusion_matrix.png`: Confusion matrix for Logistic Regression model.
- `gps_spoofing_events.csv`: Extracted GPS spoofing events.
- `gps_spoofing_log.txt`: Forensic log for GPS spoofing detection.
- `rf_jamming_events.csv`: Extracted RF jamming events.
- `rf_jamming_log.txt`: Forensic log for RF jamming detection.
- `gps_drift_plot.png`: Visualization of GPS drift with spoofing threshold.
- `dataset_sha256.txt`: SHA256 integrity hash values for generated datasets.
- `forensic_timeline.png`: Visual timeline of GPS spoofing and RF jamming events.
- `forensic_timeline_log.txt`: Log file summarising timeline generation.


### Other
- `requirements.txt`: Python dependencies required to run scripts.
- `README.md`: Project documentation.


For academic research only – © 2025 Hakan Emre Erolan, National College of Ireland.

## Usage
```bash
pip install -r requirements.txt
python generate_dataset.py

