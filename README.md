# ECG Beat Classification - MIT-BIH Preprocessing

This repository contains a preprocessing pipeline for the **MIT-BIH Arrhythmia Database**.  
The raw signals (CSV + annotation TXT files) are filtered, normalized, and segmented into individual heartbeats around R?peaks.  
The result is a compressed `.npz` file ready for training machine learning models for heartbeat classification according to the **AAMI** standard (5 classes: N, S, V, F, Q).

## Project Structure
├── mitbih_database/ # Folder containing CSV and annotation TXT files

├── src/main.ipynb # Jupyter notebook with the full preprocessing pipeline

├── models/mitbih_preprocessed.npz # Output file after running the notebook

├──README.md # This file

├── requirements.txt # Python dependencies (see below)


## Requirements

Create a virtual environment and install the required packages:


# On Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# On macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data Preparation
Download the MIT-BIH Arrhythmia Database
You need the CSV version of the database (e.g., from PhysioNet or converted from the original .dat/.hea/.atr files).
Place all .csv and annotations.txt files inside the mitbih_database/ folder.

Expected file naming
For each record (e.g., 100) there should be:

100.csv - signal data (columns: sample number, channel1, channel2)

100annotations.txt - R-peak annotations (format: sample index + symbol)

# Usage
Open and run the Jupyter notebook:


jupyter notebook main.ipynb

# The notebook will:

Load all records listed in the RECORDS variable.

# For each record:

Read the ECG signal (first channel) and convert ADC units to millivolts.

Load R?peak positions and annotation symbols.

Apply band?pass filter (0.5?40 Hz) and notch filter (50 Hz) to remove baseline wander and powerline noise.

Z?normalize the signal.

Segment heartbeats using a window of [R?100, R+150] samples (250 samples total, 360?Hz ? ~694?ms).

Map the original annotation symbols to one of the 5 AAMI classes:

N - Normal beat

S - Supraventricular ectopic beat

V - Ventricular ectopic beat

F - Fusion beat

Q - Unknown / non?beats

Concatenate all beats, labels, and patient IDs.

Display the class distribution and some example segments.

Save the data into a compressed file mitbih_preprocessed.npz.

Output File
mitbih_preprocessed.npz contains three arrays:

Key	    Shape	            Description
X	    (N_beats, 250)	    Normalized ECG segments (float32)
y	    (N_beats,)	        Integer class labels (0: N, 1: S, 2: V, 3: F, 4: Q)
pids	(N_beats,)	        Patient (record) ID for each beat (int32)

# Example loading in Python:

import numpy as np
data = np.load('mitbih_preprocessed.npz')
X, y, pids = data['X'], data['y'], data['pids']


# Notes

The original PhysioNet .dat/.hea/.atr files can be converted using the wfdb library or other tools.

The filtering parameters (cutoff frequencies, notch filter) are tuned for the MIT-BIH sampling rate (360?Hz).

The AAMI mapping follows the standard recommendation for arrhythmia classification.
