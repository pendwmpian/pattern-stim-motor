import numpy as np
import cv2
import os
# 1. Modified import path
from custom_pattern_exposure import run_alignment_and_validate
# 2. Modified import path
from pattern_on_the_fly import PatternOnTheFly # Assuming this is the correct import

# --- 1. Define DMD and Camera constants ---
DMD_WIDTH = 1920
DMD_HEIGHT = 1080
CAM_WIDTH = 1440
CAM_HEIGHT = 1080

def create_sample_canvas(canvas_size):
    """
    Creates a sample pattern on a canvas of a given size.
    The pattern consists of four 100x100 squares.
    
    Args:
        canvas_size (int): The side length of the square canvas.
        
    Returns:
        np.ndarray: A (canvas_size, canvas_size) uint8 array (0s and 255s).
    """
    print(f"Creating canvas with size: {canvas_size}x{canvas_size} pixels")
    
    # Create a blank (black) canvas
    # Use np.uint8 as this is what cv2 and the DMD will expect
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # Define properties for the squares
    square_size = 100
    gap = 50 # Gap from the center
    center = canvas_size // 2
    color = 255 # White
    thickness = -1 # Filled square
    
    # Calculate coordinates for four squares
    # (top-left x, top-left y)
    tl_x = center - gap - square_size
    tl_y = center - gap - square_size
    
    tr_x = center + gap
    tr_y = center - gap - square_size
    
    bl_x = center - gap - square_size
    bl_y = center + gap
    
    br_x = center + gap
    br_y = center + gap

    # Draw the four 100x100 squares
    # Top-left square
    cv2.rectangle(canvas, (tl_x, tl_y), (tl_x + square_size, tl_y + square_size), color, thickness)
    # Top-right square
    cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + square_size, tr_y + square_size), color, thickness)
    # Bottom-left square
    #cv2.rectangle(canvas, (bl_x, bl_y), (bl_x + square_size, bl_y + square_size), color, thickness)
    # Bottom-right square
    cv2.rectangle(canvas, (br_x, br_y), (br_x + square_size, br_y + square_size), color, thickness)
    
    print(f"Drew four 100x100 squares on the {canvas_size}x{canvas_size} canvas.")
    return canvas


def transform2DMD(canvas, M_cam_to_dmd, top_left_x, top_left_y):
    """
    Creates a 1440x1080 camera frame, "pastes" the canvas onto it,
    transforms this frame to DMD coordinates, saves it as a PNG,
    converts it to binary (0/1), and projects it.
    
    Args:
        canvas (np.ndarray): The pattern canvas (0s and 255s).
        M_cam_to_dmd (np.ndarray): The 3x3 transformation matrix (Cam -> DMD).
        top_left_x (float): The x-coordinate to paste the canvas.
        top_left_y (float): The y-coordinate to paste the canvas.
    """

    cam_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)
    canvas_size_y, canvas_size_x = canvas.shape
    
    # Calculate integer coordinates for pasting
    x_start = int(round(top_left_x))
    y_start = int(round(top_left_y))
    x_end = x_start + canvas_size_x
    y_end = y_start + canvas_size_y
    
    # Ensure paste coordinates are within camera frame bounds
    # (Simple slicing handles this well if it's within bounds)
    # Note: This assumes tx, ty are positive and within frame
    if x_start < 0 or y_start < 0 or x_end > CAM_WIDTH or y_end > CAM_HEIGHT:
        
        # Define the slices for cam_frame (clipped)
        cam_y_min = max(0, y_start)
        cam_y_max = min(CAM_HEIGHT, y_end)
        cam_x_min = max(0, x_start)
        cam_x_max = min(CAM_WIDTH, x_end)
        
        # Define the slices for the canvas (to match the clipped cam)
        can_y_min = max(0, -y_start)
        can_y_max = canvas_size_y - max(0, y_end - CAM_HEIGHT)
        can_x_min = max(0, -x_start)
        can_x_max = canvas_size_x - max(0, x_end - CAM_WIDTH)
        
        # Perform the clipped paste
        if can_y_min < can_y_max and can_x_min < can_x_max:
             cam_frame[cam_y_min:cam_y_max, cam_x_min:cam_x_max] = canvas[can_y_min:can_y_max, can_x_min:can_x_max]
    else:
        cam_frame[y_start:y_end, x_start:x_end] = canvas
    
    try:
        M_dmd_to_cam = np.linalg.inv(M_cam_to_dmd)
        print("Calculated inverse (DMD -> Cam) matrix.")
    except np.linalg.LinAlgError:
        print("ERROR: The transformation matrix is singular, cannot invert.")
        print("Cannot proceed with warping.")
        return

    M_2x3_inverse = M_dmd_to_cam[0:2, :]
    
    pattern_array_u8 = cv2.warpAffine(
        cam_frame,
        M_2x3_inverse,
        (DMD_WIDTH, DMD_HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    pattern_array_u8 = cv2.flip(pattern_array_u8, 1)
    pattern_array_binary = (pattern_array_u8 > 0).astype(np.uint8)
    
    return pattern_array_binary


def create_and_project_pattern():
    """
    Main orchestration function.
    Runs the alignment validation, then creates a sample pattern
    on the canvas, transforms it to DMD coordinates, and projects it.
    """

    M_cam_to_dmd, canvas_diameter, tx, ty, is_valid = run_alignment_and_validate()
    
    if not is_valid:
        print("Alignment not validated. Exiting pattern projection.")
        return
    
    canvas_size = int(np.ceil(canvas_diameter))
    canvas = create_sample_canvas(canvas_size)

    pattern_array_binary = transform2DMD(canvas, M_cam_to_dmd, tx, ty)

    with PatternOnTheFly(w=DMD_WIDTH, h=DMD_HEIGHT, test=False) as dmd:
        print("DMD opened. Projecting pattern...")
        # Use the binary (0/1) array here
        dmd.DefinePattern(0, exposure=1500000, darktime=0, data=pattern_array_binary)
        # 3. Changed nRepeat to 0
        dmd.SendImageSequence(nPattern=1, nRepeat=0)
        dmd.StartRunning()
        print("Pattern is running. Press Enter to stop...")
        input() # Wait for user to press Enter
        # dmd.StopRunning() # Assuming the 'with' statement handles cleanup
        print("Pattern stopped.")


if __name__ == "__main__":
    create_and_project_pattern()