
import unittest
from src.stippling import generate_seed_points
from src.image_processing import load_image

class TestStippling(unittest.TestCase):
    def test_generate_seed_points(self):
        image = load_image(r'../images/bhia.png')
        points = generate_seed_points(image, num_points=500)
        self.assertEqual(points.shape[0], 500)

if __name__ == '__main__':
    unittest.main()
