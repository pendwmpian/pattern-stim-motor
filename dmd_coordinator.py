import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image
import numpy as np
import cv2
from align_fov import get_alignment_transform

from pattern_on_the_fly import PatternOnTheFly # Assuming this is the correct import

# --- 1. Define DMD and Camera constants ---
DMD_WIDTH = 1920
DMD_HEIGHT = 1080


class DMDCoordinator:
    """
    A class to handle DMD alignment and canvas coordinate transformation.
    """
    
    # Define Constants
    DMD_WIDTH = 1920
    DMD_HEIGHT = 1080
    CAM_WIDTH = 1440
    CAM_HEIGHT = 1080

    def __init__(self):
        """
        Initializes the coordinator by running the alignment and validation process.
        """
        print("--- Initializing Coordinator and Running Alignment ---")
        
        # Call Validation and Store Results
        (self.M_2x3_inverse, 
         self.canvas_diameter, 
         self.tx, 
         self.ty, 
         self.is_valid) = self._run_alignment_and_validate()
         
        if self.is_valid:
            # Store the canvas size for convenience
            self.canvas_size = int(np.ceil(self.canvas_diameter * np.sqrt(2)))
            print("Alignment successful. Coordinator is valid and ready.")
        else:
            self.canvas_size = 0
            print("Alignment failed or was rejected. Coordinator is NOT valid.")

    def _run_alignment_and_validate(self):
        """
        Private method to run alignment.
        
        Returns:
            (matrix, float, float, float, bool): 
              (M_cam_to_dmd, circle_diameter, tx, ty, True) if validated.
              (None, None, None, None, False) if user clicked "No".
        """
        
        try:
            transform_matrix, center_x, center_y, circle_diameter = get_alignment_transform()
        except Exception as e:
             print(f"Error during 'get_alignment_transform': {e}")
             return None, None, None, None, False
        
        image_path = os.path.join("img", "align_fov", "result_with_vertices.png")
        try:
            img = Image.open(image_path)
            img.show()
        except Exception:
            print(f"Warning: Could not open result image at {image_path}")
            pass # Continue anyway

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        validation_response = messagebox.askyesno(
            title="Alignment Validation",
            message="Is the alignment shown in the image correct?"
        )
        root.destroy()
        
        if validation_response:
            # --- FIX for 2x3 matrix ---
            # Promote the 2x3 matrix to a 3x3 invertible matrix
            # by appending [0, 0, 1]
            M_cam_to_dmd = np.append(transform_matrix, [[0.0, 0.0, 1.0]], axis=0)

            try:
                # We need the inverse matrix for warpAffine (DMD -> Cam)
                M_dmd_to_cam = np.linalg.inv(M_cam_to_dmd)
            except np.linalg.LinAlgError:
                print("ERROR: The transformation matrix is singular, cannot invert.")
                print("Cannot proceed with warping.")
                return None # Return None to indicate failure

            M_2x3_inverse = M_dmd_to_cam[0:2, :]
            
            tx = center_x - circle_diameter / 2
            ty = center_y - circle_diameter / 2
            
            return M_2x3_inverse, circle_diameter, tx, ty, True
        else:
            print("User rejected the alignment.")
            return None, None, None, None, False

    def transform2DMD(self, canvas):
        """
        Class method to transform a canvas using the stored alignment.
        
        Args:
            canvas (np.ndarray): The pattern canvas (0s and 255s).
            
        Returns:
            np.ndarray: A binary (0/1) array for the DMD, or None on failure.
        """
        if not self.is_valid:
            print("ERROR: Coordinator is not valid. Cannot transform pattern.")
            return None
            
        cam_frame = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH), dtype=np.uint8)
        
        try:
            canvas_size_y, canvas_size_x = canvas.shape
        except ValueError:
            print("Error: Input canvas is not a 2D numpy array.")
            return None

        x_start = int(round(self.tx))
        y_start = int(round(self.ty))
        x_end = x_start + canvas_size_x
        y_end = y_start + canvas_size_y
        
        # --- Robust Pasting with Clipping ---
        cam_y_min = max(0, y_start)
        cam_y_max = min(self.CAM_HEIGHT, y_end)
        cam_x_min = max(0, x_start)
        cam_x_max = min(self.CAM_WIDTH, x_end)
        
        can_y_min = max(0, -y_start)
        can_y_max = canvas_size_y - max(0, y_end - self.CAM_HEIGHT)
        can_x_min = max(0, -x_start)
        can_x_max = canvas_size_x - max(0, x_end - self.CAM_WIDTH)
        
        if can_y_min < can_y_max and can_x_min < can_x_max:
             cam_frame[cam_y_min:cam_y_max, cam_x_min:cam_x_max] = canvas[can_y_min:can_y_max, can_x_min:can_x_max]
        else:
             print("Warning: Canvas was completely outside the camera frame.")
        # --- End of Pasting ---
        
        pattern_array_u8 = cv2.warpAffine(
            cam_frame,
            self.M_2x3_inverse,
            (self.DMD_WIDTH, self.DMD_HEIGHT),
            flags=cv2.INTER_NEAREST, # Use nearest neighbor for binary patterns
            borderValue=0            # Fill outside with black
        )

        # Apply horizontal flip (flip around y-axis)
        pattern_array_u8 = cv2.flip(pattern_array_u8, 1)
        
        # Convert to binary (0/1) for the DMD
        pattern_array_binary = (pattern_array_u8 > 0).astype(np.uint8)
        
        return pattern_array_binary

    def get_canvas_size(self):
        """
        Returns the calculated canvas size.
        
        Returns:
            int: The side length of the square canvas, or 0 if not valid.
        """
        return self.canvas_size


if __name__ == "__main__":
    """
    Main execution block to test the DMDCoordinator class.
    """
    
    coordinator = DMDCoordinator()
    
    if coordinator.is_valid:
        
        canvas_size = coordinator.get_canvas_size()
        print(f"\n--- Canvas size from method: {canvas_size} ---")

        print("\n--- Creating a Test Canvas ---")
        test_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cv2.rectangle(test_canvas, (0, 0), (canvas_size, canvas_size), 255, -1)
        print(f"Created a {canvas_size}x{canvas_size} test canvas with a 100x100 square.")

        print("\n--- Transforming Pattern for DMD ---")
        transformed_pattern = coordinator.transform2DMD(test_canvas)
        
        if transformed_pattern is not None:
            cv2.imwrite("warped_dmd_pattern.png", transformed_pattern * 255)
            print("Saved warped_dmd_pattern.png")
            print("Transformation successful.")
            with PatternOnTheFly(w=DMD_WIDTH, h=DMD_HEIGHT, test=False) as dmd:
                print("DMD opened. Projecting pattern...")
                # Use the binary (0/1) array here
                dmd.DefinePattern(0, exposure=1500000, darktime=0, data=transformed_pattern)
                # 3. Changed nRepeat to 0
                dmd.SendImageSequence(nPattern=1, nRepeat=0)
                dmd.StartRunning()
                print("Pattern is running. Press Enter to stop...")
                input() # Wait for user to press Enter
                dmd.StopRunning()
                print("Pattern stopped.")
        else:
            print("Pattern transformation failed. Exiting.")
    else:
        print("DMD Coordinator setup failed. Exiting.")