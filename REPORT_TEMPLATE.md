# IoT attack detection report

Replace each bracketed instruction with your own measurements after running the project. Do not present synthetic test results as dataset results.

## Objective and dataset

Describe the problem, Bot-IoT source, 5% subset and download date. Include the source citation listed on the official UNSW page. Report the actual source counts and sampled counts from `results/data_profile.json`.

## Data preparation and methodology

Describe the sampled feature columns, excluded identifiers and labels, duplicate and conflict handling, stratified split, training-only preprocessing and class imbalance. Explain that sampling occurs before splitting and the resulting test distribution does not represent the original dataset. Discuss possible correlations between flows in different splits.

## Model comparison

Insert the five model rows from `results/comparison.csv`; explain precision, recall, macro F1 and weighted F1. Include per-class support from each report. Explain how the selected model was chosen on validation data and compare it with the majority-class baseline.

## Error analysis

Insert the normalized confusion matrix for the selected model. Identify the largest confusions using the actual counts and rates. Discuss normal traffic flagged as an attack and attacks classified as normal. State the number of benign samples when interpreting the false-positive rate.

## Performance and deployment

Document the hardware and runtime from `results/benchmark_environment.json`, latency and model bundle size. The reported RSS is a process snapshot and excludes neither all unrelated memory nor captures short peaks. Explain that feature extraction, packet capture, runtime dependencies, and edge-device measurements are not included. A real Raspberry Pi or other edge-device experiment remains future work.

## Limitations and conclusion

Explain lab-origin data, class imbalance, category-capped sampling, exact-duplicate handling, the row-based split, and the absence of real-world deployment testing. State whether the feature set could be produced on the intended device. Summarize the observed tradeoff without claiming a guarantee of security.

## Contributions

Describe what each team member actually contributed and disclose any AI assistance according to your institution's rules.
