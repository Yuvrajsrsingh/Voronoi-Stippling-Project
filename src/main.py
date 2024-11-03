
from image_processing import load_image
from stippling import lloyds_algorithm
from rendering import render_stippled_image

# Load and preprocess image
image_path = r'../images/bhia.png'
image = load_image(image_path)

# Apply Lloyd's algorithm to get the final points
points = lloyds_algorithm(image, num_points=2000, iterations=50)

# Render and save the stippled image
output_path='stippled_image.png'
stippled_image = render_stippled_image(points, image, output_path)
