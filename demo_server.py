"""Local web demonstration that replays held-out dataset records."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = ('decision_tree', 'random_forest', 'logistic_regression', 'xgboost', 'dense_network')
LOCK = threading.Lock()


def create_handler(data, bundles):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload, content_type='application/json; charset=utf-8'):
            content = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                return self.send(200, (ROOT / 'demo' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if parsed.path == '/api/info':
                return self.send(200, {'models': list(bundles), 'rows': len(data), 'mode': 'held-out dataset replay'})
            if parsed.path != '/api/predict':
                return self.send(404, {'error': 'Not found'})
            query = parse_qs(parsed.query)
            model = query.get('model', ['random_forest'])[0]
            try:
                row = int(query.get('row', ['0'])[0])
            except ValueError:
                return self.send(400, {'error': 'Row must be an integer'})
            if model not in bundles or row < 0 or row >= len(data):
                return self.send(400, {'error': 'Invalid model or row'})
            bundle = bundles[model]
            sample = data.iloc[[row]][bundle['features']]
            try:
                with LOCK:
                    start = time.perf_counter()
                    prediction = bundle['pipeline'].predict(sample)
                    elapsed = (time.perf_counter() - start) * 1000
                    label = bundle['encoder'].inverse_transform(prediction.astype(int))[0]
                return self.send(200, {'row': row, 'prediction': str(label), 'actual': str(data.iloc[row]['target']), 'elapsed_ms': round(elapsed, 2), 'model': model})
            except Exception as exc:
                print('Prediction error:', repr(exc), flush=True)
                return self.send(500, {'error': 'Prediction failed; check server logs and package versions.'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--data', default=str(ROOT / 'data' / 'processed' / 'test.parquet'))
    parser.add_argument('--models', default=str(ROOT / 'models'))
    args = parser.parse_args()
    data = pd.read_parquet(args.data)
    if data.empty or 'target' not in data:
        raise SystemExit('Replay data must contain rows and a target column.')
    bundles = {}
    for name in NAMES:
        path = Path(args.models) / (name + '.joblib')
        if path.exists():
            bundle = joblib.load(path)  # Only use files trained by you or your team.
            missing = set(bundle['features']) - set(data.columns)
            if missing:
                raise SystemExit(f'{name}: missing feature columns: {sorted(missing)}')
            bundles[name] = bundle
    if not bundles:
        raise SystemExit('No trained model bundles found in ' + args.models)
    print(f'Replaying {len(data)} held-out rows with {len(bundles)} models at http://{args.host}:{args.port}', flush=True)
    ThreadingHTTPServer((args.host, args.port), create_handler(data, bundles)).serve_forever()


if __name__ == '__main__':
    main()
