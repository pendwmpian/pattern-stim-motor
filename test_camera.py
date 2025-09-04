# main.py

import cv2
# Import only the new controller class
from thorlabs_cam import ThorlabsCameraController

def main():
    """
    Main function to run the live view.
    """
    print("Application starting...")
    controller = None
    try:
        # Create the controller. All setup happens here automatically.
        # The 'with' statement ensures that controller.close() is called on exit.
        with ThorlabsCameraController(camera_index=0) as controller:
            
            print("Live view running. Press 'q' to quit.")
            
            while True:
                # The only function you need to call to get a frame!
                image_np = controller.get_nowait()

                # If a new frame is available, display it
                if image_np is not None:
                    # For color cameras, the data is RGB. OpenCV needs BGR.
                    if len(image_np.shape) == 3:
                        display_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    else:
                        display_image = image_np

                    cv2.imshow('Thorlabs Camera Live View', display_image)

                # Check for the 'q' key to quit
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Quit key pressed.")
                    break
                    
    except (ConnectionError, IndexError) as e:
        print(f"ERROR: Could not start camera. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up the OpenCV window
        cv2.destroyAllWindows()
        print("Application finished.")


if __name__ == "__main__":
    main()