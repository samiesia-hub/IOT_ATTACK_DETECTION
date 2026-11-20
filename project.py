"""Reproducible, bounded-memory IoT traffic classification prototype."""
import argparse
import json
import os
import time
import hashlib
import platform
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import psutil

DATA = Path('data/processed')
MODELS = Path('models')
RESULTS = Path('results')
SEED = 42
EXCLUDE = {'attack', 'category', 'subcategory', 'pkSeqID', 'seq', 'saddr', 'daddr', 'stime', 'ltime', 'row_id', 'Unnamed: 0'}


def dirs():
    for path in (DATA, MODELS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)


def prepare(args):
    dirs()
    files = sorted(Path(args.input).rglob('*.csv'))
    if not files:
        raise SystemExit('No CSV files found in ' + args.input)
    rng = np.random.default_rng(SEED)
    samples, counts = {}, {}
    for path in files:
        for chunk in pd.read_csv(path, chunksize=50000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            if not {'attack', 'category', 'subcategory'}.issubset(chunk.columns):
                raise SystemExit(f'{path} is missing attack/category/subcategory columns; check the official release schema.')
            attack = pd.to_numeric(chunk['attack'], errors='coerce')
            if not attack.isin([0, 1]).all() or chunk.loc[attack == 1, 'category'].isna().any():
                raise ValueError('Invalid attack labels found')
            labels = np.where(attack == 0, 'Normal', chunk['category'].astype(str).str.strip())
            chunk = chunk.assign(target=labels)
            for label, group in chunk.groupby('target'):
                counts[label] = counts.get(label, 0) + len(group)
                # Random priorities enable uniform bounded sampling across all chunks.
                group = group.copy()
                group['_priority'] = rng.random(len(group))
                combined = pd.concat([samples.get(label, group.iloc[:0]), group], ignore_index=True)
                samples[label] = combined.nlargest(args.per_class, '_priority')
    df = pd.concat(samples.values(), ignore_index=True).drop(columns='_priority')
    features = [c for c in df.columns if c not in EXCLUDE | {'target'}]
    df = df[features + ['target']].replace([np.inf, -np.inf], np.nan)
    fingerprints = pd.util.hash_pandas_object(df[features], index=False)
    conflicts = df.assign(_hash=fingerprints).groupby('_hash')['target'].nunique()
    bad = set(conflicts[conflicts > 1].index)
    df = df.loc[~fingerprints.isin(bad)].copy()
    df = df.drop_duplicates(subset=features)
    support = df.target.value_counts()
    if (support < 4).any():
        raise SystemExit('Insufficient class support after deduplication: ' + str(support.to_dict()))
    train, rest = train_test_split(df, test_size=.30, stratify=df.target, random_state=SEED)
    val, test = train_test_split(rest, test_size=.5, stratify=rest.target, random_state=SEED)
    for name, part in [('train', train), ('validation', val), ('test', test)]:
        part.to_parquet(DATA / (name + '.parquet'), index=False)
    sources = []
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        sources.append({'file': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    profile = {'sources': sources, 'source_counts': counts, 'sample_counts_after_dedup': support.to_dict(), 'split_counts': {name: part.target.value_counts().to_dict() for name, part in [('train', train), ('validation', val), ('test', test)]}, 'conflicting_fingerprints': len(bad), 'features': features, 'split': 'stratified row 70/15/15', 'seed': SEED, 'sampling': f'up to {args.per_class} rows per category before splitting'}
    (RESULTS / 'data_profile.json').write_text(json.dumps(profile, indent=2), encoding='utf8')
    print(json.dumps(profile, indent=2))


def preprocessor(X, scale):
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X if c not in numeric]
    num = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        num.append(('scale', StandardScaler()))
    cat = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encode', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('num', Pipeline(num), numeric), ('cat', cat, categorical)])


def train(args):
    dirs()
    train_df = pd.read_parquet(DATA / 'train.parquet')
    val = pd.read_parquet(DATA / 'validation.parquet')
    X, y = train_df.drop(columns='target'), train_df.target
    encoder = LabelEncoder().fit(y)
    y_enc = encoder.transform(y)
    candidates = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'decision_tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, class_weight='balanced', random_state=SEED),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=2, random_state=SEED),
        'logistic_regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=SEED),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1, tree_method='hist', n_jobs=2, random_state=SEED),
        'dense_network': MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=256, max_iter=20, random_state=SEED),
    }
    scores = {}
    for name, estimator in candidates.items():
        print('Training', name, flush=True)
        pipe = Pipeline([('preprocess', preprocessor(X, name in {'logistic_regression', 'dense_network'})), ('model', estimator)])
        start = time.perf_counter()
        pipe.fit(X, y_enc)
        fit_seconds = time.perf_counter() - start
        pred = pipe.predict(val.drop(columns='target'))
        score = f1_score(encoder.transform(val.target), pred, average='macro', zero_division=0)
        scores[name] = {'validation_macro_f1': score, 'fit_seconds': fit_seconds}
        joblib.dump({'pipeline': pipe, 'encoder': encoder, 'features': list(X.columns)}, MODELS / (name + '.joblib'))
    winner = max((k for k in scores if k != 'dummy'), key=lambda k: scores[k]['validation_macro_f1'])
    (RESULTS / 'validation.json').write_text(json.dumps({'scores': scores, 'selected_model': winner}, indent=2), encoding='utf8')
    print('Selected on validation:', winner)


def evaluate(args):
    dirs()
    df = pd.read_parquet(DATA / 'test.parquet')
    rows = []
    model_names = json.loads((RESULTS / 'validation.json').read_text(encoding='utf8'))['scores']
    for name in model_names:
        path = MODELS / (name + '.joblib')
        bundle = joblib.load(path)
        X = df[bundle['features']]
        y = bundle['encoder'].transform(df.target)
        pred = bundle['pipeline'].predict(X)
        report = classification_report(y, pred, target_names=bundle['encoder'].classes_, output_dict=True, zero_division=0)
        (RESULTS / (path.stem + '_report.json')).write_text(json.dumps(report, indent=2), encoding='utf8')
        cm = confusion_matrix(y, pred, labels=np.arange(len(bundle['encoder'].classes_)))
        pd.DataFrame(cm, index=bundle['encoder'].classes_, columns=bundle['encoder'].classes_).to_csv(RESULTS / (path.stem + '_confusion.csv'))
        normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
        pd.DataFrame(normalized, index=bundle['encoder'].classes_, columns=bundle['encoder'].classes_).to_csv(RESULTS / (path.stem + '_confusion_normalized.csv'))
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(normalized, annot=True, fmt='.2f', xticklabels=bundle['encoder'].classes_, yticklabels=bundle['encoder'].classes_, ax=ax, cmap='Blues')
        ax.set(xlabel='Predicted', ylabel='True', title=path.stem + ' normalized confusion matrix')
        fig.tight_layout()
        fig.savefig(RESULTS / (path.stem + '_confusion.png'), dpi=150)
        plt.close(fig)
        normal = bundle['encoder'].transform(['Normal'])[0] if 'Normal' in bundle['encoder'].classes_ else None
        benign = y == normal
        attack_recall = float(np.mean(pred[~benign] != normal)) if normal is not None and (~benign).any() else None
        benign_fpr = float(np.mean(pred[benign] != normal)) if normal is not None and benign.any() else None
        validation = pd.read_parquet(DATA / 'validation.parquet')
        sample = validation[bundle['features']].iloc[:min(1000, len(validation))]
        bundle['pipeline'].predict(sample.iloc[:1])
        timings = []
        for i in range(len(sample)):
            start = time.perf_counter_ns()
            bundle['pipeline'].predict(sample.iloc[i:i+1])
            timings.append((time.perf_counter_ns() - start) / 1e6)
        rows.append({'model': path.stem, 'accuracy': accuracy_score(y, pred), 'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'], 'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score'], 'attack_recall': attack_recall, 'benign_fpr': benign_fpr, 'p50_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)), 'artifact_mib': path.stat().st_size / 1048576, 'process_rss_mib': psutil.Process().memory_info().rss / 1048576})
    pd.DataFrame(rows).to_csv(RESULTS / 'comparison.csv', index=False)
    (RESULTS / 'benchmark_environment.json').write_text(json.dumps({'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': psutil.cpu_count(), 'ram_gib': psutil.virtual_memory().total / 1024**3, 'note': 'Single-row preprocessing plus model prediction on validation inputs; current process RSS is not peak memory. Packet capture and feature extraction excluded.'}, indent=2), encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False))


def predict(args):
    selected = json.loads((RESULTS / 'validation.json').read_text(encoding='utf8'))['selected_model']
    bundle = joblib.load(MODELS / (selected + '.joblib'))
    df = pd.read_csv(args.input)
    missing = set(bundle['features']) - set(df.columns)
    if missing:
        raise SystemExit('Missing required features: ' + ', '.join(sorted(missing)))
    pred = bundle['pipeline'].predict(df[bundle['features']])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'prediction': bundle['encoder'].inverse_transform(pred.astype(int))}).to_csv(args.output, index=False)
    print('Saved', args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('prepare'); p.add_argument('--input', required=True); p.add_argument('--per-class', type=int, default=100000); p.set_defaults(func=prepare)
    p = sub.add_parser('train'); p.set_defaults(func=train)
    p = sub.add_parser('evaluate'); p.set_defaults(func=evaluate)
    p = sub.add_parser('predict'); p.add_argument('--input', required=True); p.add_argument('--output', required=True); p.set_defaults(func=predict)
    args = parser.parse_args(); args.func(args)


if __name__ == '__main__':
    main()
