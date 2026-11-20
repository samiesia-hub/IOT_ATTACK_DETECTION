# IoT attack detection

This project compares five classifiers on the official Bot-IoT 5% dataset: a decision tree, random forest, logistic regression, XGBoost, and a small dense neural network. It produces per-class metrics, confusion matrices, and laptop CPU inference measurements. It is an experimental prototype, not a production intrusion prevention system.

## Setup

Install Python and create a virtual environment. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the 5% CSV files from the [official UNSW Bot-IoT dataset](https://research.unsw.edu.au/projects/bot-iot-dataset), extract them into `data/raw`, and check the actual schema. Do not commit raw data. The code expects the official feature names `attack`, `category`, and `subcategory`; it fails with an explanation if they are missing.

```powershell
.\.venv\Scripts\python.exe project.py prepare --input data/raw
.\.venv\Scripts\python.exe project.py train
.\.venv\Scripts\python.exe project.py evaluate
.\.venv\Scripts\python.exe project.py predict --input data/example.csv --output results/predictions.csv
```

Preparation uses chunked reads and retains all normal rows and up to a configurable number per attack category (`--per-class`, default 100000). It removes duplicate feature rows, quarantines conflicting labels, and creates one stratified 70/15/15 split. This row-level split can still contain correlated network events; results are not evidence of generalization to new devices or environments. The sample is category-capped, so its test class proportions do not represent the original dataset. Class counts are recorded in `results/data_profile.json`.

Training uses validation macro-F1 to pick the deployment candidate; evaluation is run afterward on the common test split. Preprocessing is fitted on training data only. The small dense network uses scikit-learn MLPClassifier, CPU-only. `results/` includes quality metrics, confusion matrices, and a laptop benchmark. Latency excludes packet capture and feature extraction and is not a measurement on edge hardware. Never load model files from untrusted sources, as joblib uses pickle.

The detailed team plan is in `IoT_Project_Plan_3_Days.txt`. The planning dashboard is `dist/index.html`.
