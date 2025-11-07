import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image
import numpy as np # Import numpy for matrix operations

# --- 1. Import the alignment function ---
# This is now done at the module level.
from align_fov import get_alignment_transform

def run_alignment_and_validate():
    """
    Runs 'get_alignment_transform()', gets alignment and circle parameters,
    shows the result image, and asks for user validation.
    
    If validated, it FIXES and returns the raw Cam->DMD transformation
    matrix, and the coordinates for the top-left corner of the canvas.
    
    The "canvas" is defined as:
    - ... existing code ...
    - ... existing code ...
    - ... existing code ...
    
    Assumes:
    1. 'get_alignment_transform' now returns:
       (transform_matrix, center_x, center_y, circle_diameter)
    2. 'transform_matrix' is a 3x3 numpy array (Cam -> DMD).
    
    Returns:
        (matrix, float, float, float, bool): 
          (M_cam_to_dmd, circle_diameter, tx, ty, True) if validated.
          (None, None, None, None, False) if user clicked "No".
    """
    
    # --- 1. Run the alignment function ---
    print(f"Attempting to run 'get_alignment_transform()'...")
    # Call the imported function directly and get new values
    transform_matrix, center_x, center_y, circle_diameter = get_alignment_transform() 
    
    print("Function 'get_alignment_transform()' executed successfully.")

    # --- 2. Show the result image ---
    image_path = os.path.join("img", "align_fov", "result_with_vertices.png")
    
    print(f"Opening result image: {image_path}")
    img = Image.open(image_path)
    img.show() # This will open the image in your default system viewer

    # --- 3. Ask for user validation ---
    
    # We need to create a simple tkinter root window
    # to be able to show the message box.
    root = tk.Tk()
    root.withdraw() # Hide the main (empty) window
    
    # Bring the dialog box to the front
    root.attributes("-topmost", True)
    
    print("Waiting for user validation... (Please check the dialog box)")
    
    # Show the blocking "Yes/No" message box
    validation_response = messagebox.askyesno(
        title="Alignment Validation",
        message="Is the alignment shown in the image correct?"
    )
    
    # Clean up the tkinter root window
    root.destroy()
    
    # --- 4. Calculate final matrix and return results ---
    if validation_response:
        print("User accepted the alignment.")

        transform_matrix = np.append(transform_matrix, [[0.0, 0.0, 1.0]], axis=0)

        # Calculate the top-left corner coordinates (the translation offset)
        tx = center_x - circle_diameter / 2
        ty = center_y - circle_diameter / 2
        
        # We NO LONGER calculate M_canvas_to_dmd here.
        # We return the raw (and fixed) matrix and the offsets.
        M_cam_to_dmd = transform_matrix
        
        print("Returning raw 'Cam-to-DMD' matrix and canvas offsets.")
        
        return M_cam_to_dmd, circle_diameter, tx, ty, True
    else:
        print("User rejected the alignment.")
        return None, None, None, None, False

if __name__ == "__main__":
    # This block runs only when you execute this script directly
    
    print("Starting DMD Alignment Validation...")
    
    # The matrix returned is now M_canvas_to_dmd
    cam_to_dmd_matrix, diameter, tx, ty, is_valid = run_alignment_and_validate()
    
    if is_valid:
        print("\nResult: User VALIDATED the alignment.")
        print(f"Canvas Diameter: {diameter} px")
        print(f"Canvas Top-Left (tx, ty): ({tx}, {ty})")
        print("Cam-to-DMD Transform Matrix:")
        print(cam_to_dmd_matrix)
        # Here, you would proceed with this new matrix
    else:
        print("\nResult: User REJECTED the alignment or the script failed.")
        # Here, you might re-run the alignment or exit
        
    print("Validation process finished.")