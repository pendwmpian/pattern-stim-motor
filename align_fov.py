# single_shot_align.py

import cv2
import numpy as np
import os
import time
import csv

# Import the camera controller library we created
from thorlabs_cam import ThorlabsCameraController
from pattern_on_the_fly import PatternOnTheFly

# --- Configuration Constants ---
RADIUS = 680
DMD_WIDTH = 1920
DMD_HEIGHT = 1080

# =============================================================================
#  Your provided circle detection function
# =============================================================================
def detectOuterCircle(image, radius=RADIUS):

    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    cv2.imwrite('./img/alignment/1_blurred.jpg', blurred)
    edges = cv2.Canny(blurred, threshold1=5, threshold2=8)
    cv2.imwrite('./img/alignment/2_edges.jpg', edges)
    
    circle_template = np.zeros((radius * 2, radius * 2), dtype=np.uint8)
    cv2.circle(circle_template, (radius, radius), radius, 255, 2)

    result = cv2.matchTemplate(edges, circle_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    center_x, center_y = max_loc[0] + radius, max_loc[1] + radius

    output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.imwrite('./img/alignment/3_circle_edge.jpg', output_image)
    return center_x, center_y, radius

def cropCircle(image, x, y, r):
    """Crops a square region around the circle and masks outside the circle."""
    
    x1, y1 = x - r, y - r  
    x2, y2 = x + r, y + r  
    
    cropped_image = image[y1:y2, x1:x2].copy()
    
    mask = np.zeros_like(cropped_image, dtype=np.uint8)
    center = (r, r) 
    cv2.circle(mask, center, r, (255, 255, 255), thickness=-1)
    
    masked_image = cv2.bitwise_and(cropped_image, mask)

    return masked_image

def line_intersection(line1, line2):
    """Finds the intersection of two lines."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0: return None
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    px = x1 + (t_num / den) * (x2 - x1)
    py = y1 + (t_num / den) * (y2 - y1)
    return (int(round(px)), int(round(py)))

def merge_lines_by_angle_cluster(lines, num_final_lines=3):
    """Merges detected lines into 3 main sides by clustering their angles."""
    if lines is None or len(lines) < num_final_lines: return []
    line_params = [{'line': line[0], 'angle': np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])} for line in lines]
    angle_points = np.array([[np.cos(2 * p['angle']), np.sin(2 * p['angle'])] for p in line_params], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(angle_points, num_final_lines, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    groups = [[] for _ in range(num_final_lines)]
    if labels is not None:
        for i, label in enumerate(labels): groups[label[0]].append(line_params[i]['line'])

    merged_lines = []
    for group in groups:
        if not group: continue
        points = np.vstack([[[l[0], l[1]], [l[2], l[3]]] for l in group])
        line_fit = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_fit.flatten()
        projections = np.dot(points - np.array([x0, y0]), np.array([vx, vy]))
        pt1, pt2 = tuple(points[np.argmin(projections)]), tuple(points[np.argmax(projections)])
        merged_lines.append(np.array([pt1[0], pt1[1], pt2[0], pt2[1]]))
    return merged_lines

def detect_vertices_from_edges(edges_image, color_image_for_drawing, output_path=None):
    """
    Takes a Canny edges image, finds the 3 sides of a triangle, and returns their corners.
    """
    # Use Hough Transform to detect line segments from the edges
    lines = cv2.HoughLinesP(edges_image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=30)
    
    # Merge the detected lines into exactly 3 final sides
    final_lines = merge_lines_by_angle_cluster(lines, num_final_lines=3)
    if len(final_lines) != 3:
        print("Warning: Could not detect exactly 3 sides for the triangle.")
        return None, None

    # Calculate the intersection points (corners) of the 3 lines
    corners = [
        line_intersection(final_lines[0], final_lines[1]),
        line_intersection(final_lines[1], final_lines[2]),
        line_intersection(final_lines[2], final_lines[0])
    ]
    corners = [c for c in corners if c is not None]

    if len(corners) != 3:
        print("Warning: Could not find 3 valid intersection points.")
        return None, None

    # If an output path is provided, draw the results and save the image
    if output_path:
        output_img = color_image_for_drawing.copy()
        for line in final_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for corner in corners:
            cv2.circle(output_img, corner, 10, (0, 0, 255), -1)
        cv2.imwrite(output_path, output_img)
        print(f"Saved vertex detection result to '{output_path}'")
        
    return corners, final_lines


# =============================================================================
#  Main Program Logic
# =============================================================================
def get_alignment_transform():
    """
    Initializes the camera, captures a single image, processes it to find
    the FOV circle, and displays the result.
    """
    print("Application starting...")
    os.makedirs('./img/alignment', exist_ok=True)

    with ThorlabsCameraController(camera_index=0) as controller:
        
        # --- PHASE 1: Capture initial FOV image ---
        print("\n--- Phase 1: Capturing initial FOV image ---")
        captured_image, _ = controller.get_nowait()
        while captured_image is None:
            captured_image, _ = controller.get_nowait()
            time.sleep(0.05)
        print(f"Initial image acquired. Shape: {captured_image.shape}")

        # Prepare BGR image for display and processing
        if len(captured_image.shape) == 3: # Color
            display_image = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
        else: # Monochrome
            display_image = cv2.cvtColor(captured_image, cv2.COLOR_GRAY2BGR)

        # --- PHASE 2: Detect FOV Circle ---
        print("\n--- Phase 2: Detecting FOV circle ---")
        bordered_display = cv2.copyMakeBorder(display_image, RADIUS, RADIUS, RADIUS, RADIUS, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Note: These coordinates are for the bordered image
        center_x, center_y, detected_radius = detectOuterCircle(cv2.cvtColor(bordered_display, cv2.COLOR_BGR2GRAY), radius=RADIUS)
        print(f"FOV circle detected at ({center_x}, {center_y}) on the bordered image.")
        
        # --- PHASE 3: Project the triangle pattern using the DMD ---
        print("\n--- Phase 3: Projecting triangle pattern via DMD ---")
        unique_pattern_orig = cv2.imread('unique_pattern.png', cv2.IMREAD_GRAYSCALE)
        if unique_pattern_orig is None:
            print("ERROR: Could not load 'unique_pattern.png'.")
            return
        
        unique_pattern = cv2.flip(unique_pattern_orig, 1)

        try:
            with PatternOnTheFly(w=DMD_WIDTH, h=DMD_HEIGHT, test=False) as dmd:
                print("Projecting pattern...")
                dmd.DefinePattern(0, exposure=1500000, darktime=0, data=unique_pattern)
                dmd.SendImageSequence(nPattern=1, nRepeat=1)
                dmd.StartRunning()
            print("Waiting for DMD projection to stabilize...")
            time.sleep(1)
        except Exception as e:
            print(f"ERROR: DMD Initialization or operation failed: {e}")
            return

        # --- PHASE 4: Capture the image with the projected pattern ---
        print("\n--- Phase 4: Capturing image with projected triangle ---")
        triangle_image, _ = controller.get_nowait()
        while triangle_image is None:
            triangle_image, _ = controller.get_nowait()
            time.sleep(0.05)
        print("Successfully acquired image with the projected pattern.")
        
        if len(triangle_image.shape) == 3: # Color
            triangle_image_bgr = cv2.cvtColor(triangle_image, cv2.COLOR_RGB2BGR)
        else: # Mono
            triangle_image_bgr = cv2.cvtColor(triangle_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("captured_with_triangle.png", triangle_image_bgr)

        # --- PHASE 5: Crop and Process Triangle Image ---
        print("\n--- Phase 5: Cropping and processing for triangle edges ---")
        
        # Add a border to the new image to match the coordinate system of center_x, center_y
        bordered_triangle_image = cv2.copyMakeBorder(triangle_image_bgr, RADIUS, RADIUS, RADIUS, RADIUS, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Crop the image to the FOV circle
        cropped_fov = cropCircle(bordered_triangle_image, center_x, center_y, detected_radius)
        
        # Isolate and filter the blue channel
        blue_channel = cropped_fov[:, :, 0] # BGR format, so index 0 is Blue
        _, blue_filtered = cv2.threshold(blue_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Perform Canny edge detection
        edges = cv2.Canny(blue_filtered, 1000, 1000, apertureSize=3)
        cv2.imwrite("debug_triangle_edges.png", edges)
        print("Saved the detected triangle edges as 'debug_triangle_edges.png'")

        # --- PHASE 6: Find Vertices and Calculate Transform ---
        print("\n--- Phase 6: Finding Vertices and Calculating Transform ---")
        
        # Detect vertices from the 'edges' image
        # The returned vertices are in the coordinate system of the 'cropped_fov' image
        local_vertices, _ = detect_vertices_from_edges(edges, cropped_fov, output_path='result_with_vertices.png')
        
        if local_vertices:
            # Convert local vertex coordinates back to the full, original image coordinate system
            crop_origin_x = center_x - detected_radius - RADIUS
            crop_origin_y = center_y - detected_radius - RADIUS
            
            full_image_vertices = []
            for vx, vy in local_vertices:
                full_x = vx + crop_origin_x
                full_y = vy + crop_origin_y
                full_image_vertices.append([full_x, full_y])
            print(f"Detected vertices in full image coordinates: {full_image_vertices}")

            # Load ideal vertices from CSV
            ideal_vertices = []
            with open('triangle_vertices.csv', 'r') as f:
                reader = csv.reader(f); next(reader)
                for row in reader: ideal_vertices.append([int(row[1]), int(row[2])])

            # Sort both sets of vertices to ensure correct matching for the transform
            ideal_vertices.sort(key=lambda p: p[1])
            full_image_vertices.sort(key=lambda p: p[1])

            # Calculate the Affine Transformation Matrix
            pts_ideal = np.float32(ideal_vertices)
            pts_detected = np.float32(full_image_vertices)
            M = cv2.getAffineTransform(pts_ideal, pts_detected)

            print("\n--- SUCCESS: AFFINE TRANSFORMATION MATRIX (DMD -> CAMERA) ---")
            print("This matrix maps coordinates from the DMD pattern to the camera image.")
            print(M)
            print("----------------------------------------------------------")

            return M
        
        else: return None

if __name__ == '__main__':
    get_alignment_transform()