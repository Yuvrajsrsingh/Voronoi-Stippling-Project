# **Voronoi Stippling Project**

This project implements **Weighted Voronoi Stippling**, an artistic technique that generates a stippled (dot-based) representation of an image using Voronoi diagrams and Lloyd's relaxation algorithm. The stippling is weighted based on the brightness (intensity) of the input image, ensuring denser stippling in darker areas and lighter stippling in brighter areas, mimicking traditional stippling art.

## **Overview**

Voronoi Stippling creates an artistic representation of an image by generating dots (stipples) based on an image's intensity values. Using **Lloyd's algorithm**, the stipples are iteratively refined to create even distributions of points across the image, with more points appearing in darker regions.

The image undergoes the following steps:

1. **Load Image**: The image is loaded and converted to grayscale for brightness analysis.
2. **Random Point Generation**: Points are randomly distributed, favoring darker areas with higher intensity for denser stippling.
3. **Voronoi Diagram Construction**: Voronoi diagrams are created to divide the image into regions based on the points.

4. **Lloyd's Relaxation**: Points are updated iteratively using Lloyd's relaxation algorithm, adjusting each point to the centroid of its corresponding Voronoi region. This improves point distribution and refines the stippling effect.

5. **Rendering**: The stippled image is generated, with denser stippling in darker regions. The final result is saved as an image file.

Here's an example of the process:

| ![final_otuput](images/final_output.png) |
| :--------------------------------------: |
|          Original and Stippled           |

## **Features**

- **Lloyd's Relaxation Algorithm** for Voronoi optimization.
- **Intensity-based Stippling**: Denser dots in darker areas for a more natural stippling effect.
- Outputs a stippled image in `.png` format.
- Configurable parameters such as the number of points and iterations for fine-tuning the result.
- Handles edge cases, including invalid or degenerate Voronoi cells.

## **Installation**

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Yuvrajsrsingh/Voronoi-Stippling-Project.git
   cd voronoi_stippling
   ```
