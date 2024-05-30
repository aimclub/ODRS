import unittest
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.src.data_processing.ml_processing.recommendation_module import predict_models


class TestDatasetInfo(unittest.TestCase):
    def test_dataset_info(self):
        dataset_path = "/home/runner/work/ODRS/src/user_datasets/WaRP/Warp-D"
        classes_path = "/home/runner/work/ODRS/src/classes.txt"
        run_path = '/home/runner/work/ODRS/src/'

        result = predict_models(dataset_path, classes_path, run_path)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], int)
        self.assertIsInstance(result[2], int)


if __name__ == "__main__":
    unittest.main()