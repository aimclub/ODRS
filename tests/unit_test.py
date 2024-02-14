import unittest
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.ODRS.utils.dataset_info import dataset_info


class TestDatasetInfo(unittest.TestCase):
    def test_dataset_info(self):
        dataset_path = "/home/runner/work/ODRS/ODRS/user_datasets/WaRP/Warp-D"
        classes_path = "/home/runner/work/ODRS/ODRS/classes.txt"

        result = dataset_info(dataset_path, classes_path)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

if __name__ == "__main__":
    unittest.main()