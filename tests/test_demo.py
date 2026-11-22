import unittest
from unittest.mock import Mock

import pandas as pd
from demo_server import create_handler


class DemoTest(unittest.TestCase):
    def setUp(self):
        encoder = Mock()
        encoder.inverse_transform.return_value = ['Normal']
        pipeline = Mock()
        import numpy as np
        pipeline.predict.return_value = np.array([0])
        bundle = {'features': ['rate'], 'encoder': encoder, 'pipeline': pipeline}
        handler = create_handler(pd.DataFrame({'rate': [1.0], 'target': ['Normal']}), {'random_forest': bundle})
        self.handler = object.__new__(handler)
        self.handler.send = Mock()

    def test_prediction_and_invalid_input(self):
        self.handler.path = '/api/predict?model=random_forest&row=0'
        self.handler.do_GET()
        status, result = self.handler.send.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual(result['prediction'], 'Normal')
        self.assertEqual(result['actual'], 'Normal')
        self.handler.path = '/api/predict?model=random_forest&row=99'
        self.handler.do_GET()
        self.assertEqual(self.handler.send.call_args.args[0], 400)


if __name__ == '__main__':
    unittest.main()
