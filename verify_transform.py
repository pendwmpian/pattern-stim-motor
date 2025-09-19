# verify_transform.py

import cv2
import numpy as np

# Import the function from your modified alignment script
from align_fov import get_alignment_transform

def main():
    """
    Gets the alignment matrix, transforms the original pattern,
    and overlays it on the captured image for verification.
    """
    print("--- Verification Script Started ---")

    # 1. Run the full alignment process to get the transformation matrix
    #    This will trigger the camera, DMD, and all processing.
    M = get_alignment_transform()

    if M is None:
        print("\nERROR: Could not get the transformation matrix. Exiting.")
        return

    print("\nSuccessfully obtained the transformation matrix:")
    print(M)

    # 2. Load the source and destination images
    print("\nLoading images for overlay...")
    # The original pattern you want to transform
    pattern_image = cv2.imread('./unique_pattern/unique_pattern.png') 
    # The image the pattern was projected onto
    captured_image = cv2.imread('./img/align_fov/captured_with_triangle.png')

    if pattern_image is None or captured_image is None:
        print("ERROR: Could not load 'unique_pattern.png' or 'captured_with_triangle.png'.")
        return

    # 3. Apply the affine transformation to the pattern image
    #    This will warp the pattern to fit its position in the captured image.
    h, w = captured_image.shape[:2]
    transformed_pattern = cv2.warpAffine(pattern_image, M, (w, h))
    
    # Save the transformed pattern for inspection
    cv2.imwrite('./img/align_fov/transformed_pattern.png', transformed_pattern)
    print("Saved the warped pattern as 'transformed_pattern.png'")

    # 4. Blend the captured image and the transformed pattern
    #    Use addWeighted for a 50% transparent overlay effect.
    alpha = 0.5 # Transparency of the captured image
    beta = 0.5  # Transparency of the transformed pattern
    gamma = 0   # Scalar added to each sum
    
    overlay_image = cv2.addWeighted(captured_image, alpha, transformed_pattern, beta, gamma)
    
    print("Successfully created the overlay image.")

    # 5. Save and display the final result
    cv2.imwrite('./img/align_fov/verification_overlay.png', overlay_image)
    print("Saved the final result as 'verification_overlay.png'")

    cv2.imshow('Verification Overlay', overlay_image)
    print("\nDisplaying the final overlay. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--- Verification Script Finished ---")


if __name__ == "__main__":
    main()