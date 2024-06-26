import unittest
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.src.data_processing.ml_processing.recommendation_module import predict_models
from ODRS.src.data_processing.data_utils.utils import load_class_names, get_models, get_data_path

class TestDatasetInfo(unittest.TestCase):

    def test_load_classes(self):
        classes_path = "/home/runner/work/ODRS/src/classes.txt"
        result = load_class_names(classes_path)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 28)

    
    def test_models(self):
        result = get_models()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 15)

    def test_data_path(self):
        default = '/home/runner/work/ODRS/ODRS/user_datasets/WaRP/Warp-D'
        result = get_data_path(default)
        self.assertEqual(result, default)



if __name__ == "__main__":
    unittest.main()