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

Preparation uses chunked reads and retains up to a configurable number of rows per category, including normal traffic (`--per-class`, default 100000). It removes duplicate feature rows and quarantines conflicting labels within the sampled records, then creates one stratified 70/15/15 split. This row-level split can still contain correlated network events; results are not evidence of generalization to new devices or environments. The sample is category-capped, so its test class proportions do not represent the original dataset. Source checksums and class counts are recorded in `results/data_profile.json`. The actual source feature schema and its suitability for real-time extraction must be reviewed before interpreting results.

Training uses validation macro-F1 to pick the deployment candidate; evaluation is run afterward on the common test split. Preprocessing is fitted on training data only. The small dense network uses scikit-learn MLPClassifier, CPU-only. `results/` includes per-class JSON reports, confusion matrices as CSV and PNG, and a comparison CSV. Inference timing measures preprocessing and single-row prediction on validation rows, and RSS is a snapshot of the current process, not peak memory. Latency excludes packet capture and feature extraction and is not a measurement on edge hardware. Never load model files from untrusted sources, as joblib uses pickle.

Run the end-to-end test with artificial data using `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. The test runs in a temporary directory and does not create real dataset results. After running the real experiment, use `REPORT_TEMPLATE.md` to explain the actual numbers and limitations. The example prediction command requires a CSV containing the feature columns recorded in `results/data_profile.json`.

The detailed team plan is in `IoT_Project_Plan_3_Days.txt`. The planning dashboard is `dist/index.html`.

## Ubuntu server replay demo

The demo replays held-out dataset rows and displays each prediction and its known label. It does not capture or classify live network traffic. Many Bot-IoT aggregate features require an extractor that is not included here. Keep this unauthenticated development server on localhost; use SSH tunneling rather than exposing it to the internet.

The model bundles and processed dataset are intentionally excluded from Git. On your Ubuntu server, clone this repository, then copy your trusted `models/*.joblib` files and `data/processed/test.parquet` from your Windows computer into the matching directories on the server. You may copy only the models you want to demo, including `random_forest.joblib`. These files are local to your computer and do not appear in the GitHub repository.

From Windows PowerShell, replace the placeholders with your server's SSH username, address, and absolute project path (create those directories on the server first):

```powershell
scp .\models\*.joblib USER@SERVER:/absolute/path/IOT_ATTACK_DETECTION/models/
scp .\data\processed\test.parquet USER@SERVER:/absolute/path/IOT_ATTACK_DETECTION/data/processed/
```

On the Ubuntu server, from the repository directory, use Python 3.13 to match the training environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r demo-requirements.txt
.venv/bin/python demo_server.py
```

If Python 3.13 or a compatible package wheel is unavailable for your server's CPU architecture, install a compatible Python environment or retrain in the server environment. Pickle/joblib models are not guaranteed to load across library versions; never load untrusted model files. Stop the server with Ctrl+C. No system service or network configuration is changed by these commands.

On your Windows computer, open another PowerShell window and create an SSH tunnel, then leave it open:

```powershell
ssh -L 8501:127.0.0.1:8501 USER@SERVER
```

Open `http://localhost:8501` in your browser. The interface shows predictions, actual labels, and per-row inference times. It displays at most 30 recent records; the replay position is local to each browser. Inference timings in this interactive demo are illustrative, not a controlled benchmark. To measure your server more systematically, run `python project.py evaluate` there after also copying the validation and test parquet files, model bundles, and `results/validation.json`; review the benchmark limitations above. The dataset and models remain on your own machines.
