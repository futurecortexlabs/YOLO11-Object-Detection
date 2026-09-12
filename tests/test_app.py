import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image
import app

class DetectionTests(unittest.TestCase):
    def test_no_input(self):
        result = app.detect_objects(None, .25, .45)
        self.assertIsNone(result[0])
        self.assertTrue(result[1].empty)
        self.assertIsNone(result[-1])

    def test_detection_rgb_csv_and_resize(self):
        box = MagicMock()
        box.cls = [np.array(0)]
        box.conf = [np.array(.95)]
        box.xyxy = [np.array([1, 2, 30, 40])]
        result = MagicMock()
        result.names = {0: 'person'}
        result.boxes = [box]
        result.plot.return_value = np.array([[[0, 0, 255]]], dtype=np.uint8)
        model = MagicMock()
        model.predict.return_value = [result]
        with patch('app.load_model', return_value=model):
            out = app.detect_objects(Image.new('RGB', (2400, 1200)), .25, .45)
        self.assertEqual(out[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(out[3:5], ('1', '1'))
        self.assertEqual(model.predict.call_args.kwargs['source'].size, (1920, 960))
        self.assertIn('1920', out[2])
        self.assertTrue(Path(out[-1]).read_bytes().startswith(b'\xef\xbb\xbf'))
        Path(out[-1]).unlink()

    def test_no_detections(self):
        result = MagicMock()
        result.boxes = []
        result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        model = MagicMock()
        model.predict.return_value = [result]
        with patch('app.load_model', return_value=model):
            out = app.detect_objects(Image.new('RGB', (10, 10)), .25, .45)
        self.assertEqual(out[3:5], ('0', '0'))
        self.assertEqual(list(out[1].columns), app.COLUMNS)
        Path(out[-1]).unlink()

    def test_failure_clears_results(self):
        with patch('app.load_model', side_effect=RuntimeError('private detail')):
            out = app.detect_objects(Image.new('RGB', (10, 10)), .25, .45)
        self.assertIsNone(out[0])
        self.assertIsNone(out[-1])
        self.assertNotIn('private detail', out[2])

if __name__ == '__main__':
    unittest.main()
