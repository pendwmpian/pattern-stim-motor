import numpy as np
import cv2
import csv

DMD_WIDTH = 1920
DMD_HEIGHT = 1080
LINE_WIDTH = 20
TRIANGLE_WIDTH = 350

def generate_triangle_pattern(width, height, triangle_width, line_width):
    """
    Generates a pattern with a non-equilateral triangle.

    Args:
        width (int): The width of the pattern image.
        height (int): The height of the pattern image.
        triangle_width (int): The approximate width of the triangle.
        line_width (int): The width of the lines forming the triangle.

    Returns:
        np.ndarray: A black and white image with the generated pattern.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Define the vertices of the non-equilateral triangle
    center_x = width // 2
    center_y = height // 2
    
    half_width = triangle_width // 2
    
    # Define 3 points for the triangle
    pt1 = (center_x - half_width, center_y + int(half_width * 1.8))
    pt2 = (center_x + half_width, center_y + int(half_width * 0.5))
    pt3 = (center_x, center_y - half_width)

    vertices = [pt1, pt2, pt3]

    # Draw the lines of the triangle
    cv2.line(image, pt1, pt2, (255, 255, 255), line_width)
    cv2.line(image, pt2, pt3, (255, 255, 255), line_width)
    cv2.line(image, pt3, pt1, (255, 255, 255), line_width)

    # pts = np.array(vertices, np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # cv2.fillPoly(image, [pts], 255)

    return image, vertices

def save_vertices_to_csv(vertices, filename="triangle_vertices.csv"):
    """
    Saves the vertex coordinates to a CSV file.

    Args:
        vertices (list): A list of (x, y) coordinate tuples.
        filename (str): The name of the output CSV file.
    """
    # Define the header for the CSV file
    header = ['vertex_name', 'x', 'y']
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(header)
        
        # Write the vertex data
        for i, vertex in enumerate(vertices):
            writer.writerow([f'pt{i+1}', vertex[0], vertex[1]])

if __name__ == '__main__':
    pattern_image, vertex_coords = generate_triangle_pattern(DMD_WIDTH, DMD_HEIGHT, TRIANGLE_WIDTH, LINE_WIDTH)
    cv2.imwrite('unique_pattern.png', pattern_image)
    save_vertices_to_csv(vertex_coords)
    print("Generated 'unique_pattern.png' with a non-equilateral triangle.")
