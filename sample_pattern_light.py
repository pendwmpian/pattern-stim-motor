import numpy as np
import cv2
import os

from dmd_coordinator import DMDCoordinator
from pattern_on_the_fly import PatternOnTheFly


SQUARE_SIZE = 200
STEP_SIZE = 30    # Pixels to move per frame
NUM_FRAMES = 100  # Updated to 100 frames
EXPOSURE_TIME_MS = 20 # Exposure for ~30fps
EXPOSURE_TIME_US = EXPOSURE_TIME_MS * 1000
# How many frames to spend *only* moving Y after hitting a wall
# "a few steps" = 1 frame for this logic
Y_STEPS_ON_TURN = 5 


def run_zigzag_sequence():
    """
    Initializes the DMD coordinator, generates a 100-frame "zig-zag walking"
    pattern sequence, and projects it.
    """
    
    # --- 1. Initialize Coordinator ---
    coordinator = DMDCoordinator()
    
    if not coordinator.is_valid:
        print("DMD Coordinator setup failed. Exiting.")
        return
        
    canvas_size = coordinator.get_canvas_size()
    if canvas_size == 0:
        print("Canvas size is 0. Exiting.")
        return

    print(f"\n--- Step 1: Coordinator Valid. Canvas Size: {canvas_size}x{canvas_size} ---")

    # --- 2. Generate Frames ---
    print(f"--- Step 2: Generating {NUM_FRAMES} frames... ---")
    
    binary_frames = []
    
    # Pre-calculate movement bounds
    max_x = canvas_size - SQUARE_SIZE
    max_y = canvas_size - SQUARE_SIZE
    
    # Initial position
    current_x = 0.0
    current_y = 0.0
    
    # State machine variables
    state = 'move_x' # 'move_x' or 'move_y'
    x_direction = 1  # 1 for right, -1 for left
    y_steps_taken = 0

    for i in range(NUM_FRAMES):
        # 1. Create the blank canvas
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        
        # 2. Draw the square at the current position
        # Clamp positions to be safe
        x1 = int(round(np.clip(current_x, 0, max_x)))
        y1 = int(round(np.clip(current_y, 0, max_y)))
        x2 = x1 + SQUARE_SIZE
        y2 = y1 + SQUARE_SIZE
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, -1)
        
        # 3. Transform the canvas to a DMD binary pattern
        print(f"Transforming frame {i+1}/{NUM_FRAMES}...")
        binary_pattern = coordinator.transform2DMD(canvas)
        
        if binary_pattern is not None:
            binary_frames.append(binary_pattern)
        else:
            print(f"Warning: Frame {i+1} transformation failed. Skipping.")

        # 4. Update state and position for the *next* frame
        if state == 'move_x':
            current_x += x_direction * STEP_SIZE
            
            # Check for horizontal bounds
            if current_x > max_x:
                current_x = max_x # Clamp at edge
                state = 'move_y'  # Switch to Y-move state
                x_direction = -1  # Set next X direction
                y_steps_taken = 0 # Reset Y-step counter
            elif current_x < 0:
                current_x = 0     # Clamp at edge
                state = 'move_y'  # Switch to Y-move state
                x_direction = 1   # Set next X direction
                y_steps_taken = 0 # Reset Y-step counter
                
        elif state == 'move_y':
            current_y += STEP_SIZE
            y_steps_taken += 1
            
            # Check if Y-move is done
            if y_steps_taken >= Y_STEPS_ON_TURN:
                state = 'move_x' # Switch back to X-move state
                
            # Check for vertical bounds (stop if we hit the bottom)
            if current_y > max_y:
                current_y = max_y
                state = 'move_x' # No more room, go back to X
                print("Warning: Reached bottom of canvas.")
    
    if not binary_frames:
        print("No frames were generated. Exiting.")
        return
        
    print(f"\n--- Step 3: Projecting {len(binary_frames)}-frame sequence ---")
    
    try:
        with PatternOnTheFly(w=coordinator.DMD_WIDTH, h=coordinator.DMD_HEIGHT, test=False) as dmd:
            
            # Define all patterns in the sequence
            for i, frame_data in enumerate(binary_frames):
                dmd.DefinePattern(
                    i, 
                    exposure=EXPOSURE_TIME_US, 
                    darktime=0, 
                    data=frame_data
                )
            
            print(f"Defined {len(binary_frames)} patterns.")
            
            dmd.SendImageSequence(nPattern=len(binary_frames), nRepeat=0) # nRepeat=0 for infinite loop
            dmd.StartRunning()
            
            print("Pattern sequence is running. Press Enter to stop...")
            input() 
            print("Pattern stopped.")
            
    except Exception as e:
        print(f"\nAn error occurred during DMD projection: {e}")

if __name__ == "__main__":
    run_zigzag_sequence()