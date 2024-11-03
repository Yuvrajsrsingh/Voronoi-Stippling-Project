import cv2
import numpy as np

def render_stippled_image(points, image, output_path='stippled_image.png'):
    """Render stippled image by drawing points onto a blank canvas."""
    #Creating a canvas
    stippled_image = np.ones_like(image) * 255  
    # Draw points (stipples) on the canvas
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(stippled_image, (x, y), radius=1, color=(0), thickness=-1)  

    # Save the stippled image
    cv2.imwrite(output_path, stippled_image)
    print(f"Stippled image saved at {output_path}")

    return stippled_image
