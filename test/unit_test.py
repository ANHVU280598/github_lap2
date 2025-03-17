import unittest

from ml_code.data_load import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data('')
        self.assertIsNone(data)
        self.assertEqual(len(data), 150)

if __name__ == '__main__':
    unittest.main()