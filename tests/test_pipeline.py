"""Smoke-test the CLI with artificial data, never the real experiment."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'project.py'


class PipelineTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'raw'
            raw.mkdir()
            rng = np.random.default_rng(42)
            rows = []
            for category in ['Normal', 'DDoS', 'Recon']:
                for _ in range(100):
                    rows.append({'attack': int(category != 'Normal'), 'category': category,
                                 'subcategory': category, 'pkSeqID': len(rows),
                                 'rate': rng.random(), 'bytes': int(rng.integers(1, 10000)),
                                 'proto': rng.choice(['tcp', 'udp'])})
            pd.DataFrame(rows).to_csv(raw / 'synthetic.csv', index=False)
            env = dict(os.environ, MPLBACKEND='Agg')
            def run(*args):
                return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=root,
                                      env=env, check=True, capture_output=True, text=True)
            run('prepare', '--input', str(raw), '--per-class', '100')
            splits = [pd.read_parquet(root / 'data' / 'processed' / (name + '.parquet')) for name in ('train', 'validation', 'test')]
            for i in range(3):
                for j in range(i + 1, 3):
                    a = set(pd.util.hash_pandas_object(splits[i].drop(columns='target'), index=False))
                    b = set(pd.util.hash_pandas_object(splits[j].drop(columns='target'), index=False))
                    self.assertFalse(a & b)
            run('train')
            run('evaluate')
            run('predict', '--input', str(raw / 'synthetic.csv'), '--output', 'results/predictions.csv')
            self.assertEqual(len(pd.read_csv(root / 'results' / 'comparison.csv')), 6)
            self.assertEqual(len(pd.read_csv(root / 'results' / 'predictions.csv')), 300)
            self.assertTrue((root / 'results' / 'xgboost_confusion.png').exists())


if __name__ == '__main__':
    unittest.main()
