
import cv2
import matplotlib.pyplot as plt

def load_image(image_path, size=(1080, 560)):
    """Loads an image and converts it to grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)  # Resize for easier processing
    return image

def show_image(image, title="Image"):
    """Displays the image using matplotlib."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
