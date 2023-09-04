import unittest
import os
import sys
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(project_dir)))
from ODRS.ODRS.data_utils.dataset_info import dataset_info, process_directory_img


class TestDatasetInfo(unittest.TestCase):
    def test_dataset_info(self):
        dataset_path = "/home/runner/work/ODRS/ODRS/user_datasets/yolo/aeral"
        classes_path = "/home/runner/work/ODRS/classes.txt"

        result = dataset_info(dataset_path, classes_path)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertIsInstance(result[4], float)

    def test_process_directory_img(self):
        test_dir = "/home/runner/work/ODRS/ODRS/user_datasets/yolo/aeral/test"
        os.makedirs(test_dir, exist_ok=True)

        label_path = os.path.join(test_dir, 'labels')
        image_path = os.path.join(test_dir, 'images')
        os.makedirs(label_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)

        with open(os.path.join(image_path, 'image1.jpg'), 'w') as f:
            pass
        with open(os.path.join(image_path, 'image2.png'), 'w') as f:
            pass
        with open(os.path.join(label_path, 'label.txt'), 'w') as f:
            pass

        result = process_directory_img(test_dir)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        image_count, image_size = result
        self.assertIsInstance(image_count, int)
        self.assertIsInstance(image_size, tuple)
        self.assertEqual(len(image_size), 2)

if __name__ == "__main__":
    unittest.main()