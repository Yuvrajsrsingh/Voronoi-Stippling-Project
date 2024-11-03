import cv2 
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np
import matplotlib.pyplot as plt




def generate_seed_points(image, num_points=1000):
    """Generates random seed points across the image."""
    rows, cols = image.shape
    points = np.random.rand(num_points, 2)
    points[:, 0] *= cols  # Scale to image width
    points[:, 1] *= rows  # Scale to image height
    return points
def generate_brightness_weighted_points(image, num_points=1000):
    """Generates points weighted by image brightness (darker regions get more points)."""
    rows, cols = image.shape
    points = []
    while len(points) < num_points:
        x = np.random.randint(0, cols)
        y = np.random.randint(0, rows)
        brightness = image[y, x]
        
        # Probability threshold based on brightness
        if np.random.rand() < (1 - brightness / 255):  # Darker pixels get more points
            points.append([x, y])
    return np.array(points)


def compute_voronoi(points, image):
    """Computes the Voronoi diagram based on the points."""
    vor = Voronoi(points)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=1.5)
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()
def compute_centroids(vor, image, density_function):
    """Computes the centroids of the Voronoi regions weighted by the image intensity."""
    centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            polygon = np.array(polygon)
            if polygon.shape[0] > 2:  # Avoid degenerate polygons
                # Calculate weighted centroid
                x, y = polygon.mean(axis=0)
                centroids.append([x, y])
    return np.array(centroids)

def update_points(points, centroids):
    """Updates points by moving them towards the computed centroids."""
    return centroids


def lloyds_algorithm(image, num_points=1000, iterations=50, alpha=0.5):
    points = generate_brightness_weighted_points(image, num_points)
    
    for i in range(iterations):
        vor = Voronoi(points)
        
        centroids = compute_brightness_weighted_centroids(vor, image)
        
        print(f"Iteration {i}: Number of points: {len(points)}, Number of centroids: {len(centroids)}")
        
        if len(centroids) == 0:
            print("No centroids found. Ending iterations.")
            break
        
        # Match the number of points with the number of centroids
        if len(centroids) < len(points):
            # Randomly select from the centroids or interpolate
            new_points = np.zeros_like(points)
            for j in range(len(points)):
                if j < len(centroids):
                    new_points[j] = (1 - alpha) * points[j] + alpha * centroids[j]
                else:
                    new_points[j] = points[j]  # Keep original if no centroid exists
            points = new_points
        else:
            points = (1 - alpha) * points + alpha * centroids

        # Optional: comment this section out to speed up iterations
        if i % 20 == 0:  # Adjust frequency as needed
            voronoi_plot_2d(vor, show_vertices=False, line_colors='orange')
            plt.scatter(points[:, 0], points[:, 1], c='blue', s=1)
            plt.title(f"Lloyd's Algorithm Iteration {i}")
            plt.xlim(0, image.shape[1])
            plt.ylim(0, image.shape[0])
            plt.gca().invert_yaxis()
            plt.show()

    return points



    

def compute_brightness_weighted_centroids(vor, image):
    centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            polygon = np.array(polygon)

            if polygon.shape[0] > 2:  # Ensure valid polygon
                # Mask for the region
                mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [np.int32(polygon)], 1)

                # Calculate region intensity
                region_intensity = np.multiply(mask, 255 - image)
                sum_intensity = np.sum(region_intensity)

                if sum_intensity > 0:
                    x_coords, y_coords = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
                    centroid_x = np.sum(region_intensity * x_coords) / sum_intensity
                    centroid_y = np.sum(region_intensity * y_coords) / sum_intensity
                    centroids.append([centroid_x, centroid_y])
                else:
                    # Fallback for too bright regions
                    centroids.append(polygon.mean(axis=0))  # Geometric centroid
    return np.array(centroids)

