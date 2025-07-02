import sys
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage
import time

try:
    from thorcam.camera import ThorCam
except ImportError:
    print("Error: 'thorcam' library not found.")
    print("Please install it using: pip install thorcam")
    sys.exit(1)

class CameraManager(QObject):
    """
    A QObject class to manage a Thorlabs camera in a separate thread.
    Handles initialization, image acquisition, and cleanup.
    """
    image_acquired = pyqtSignal(np.ndarray)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    camera_initialized = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.camera = None
        self._is_initialized = False

    @pyqtSlot()
    def initialize_camera(self):
        """
        Initializes the connection to the Thorlabs camera.
        """
        if self._is_initialized:
            self.status_updated.emit("Camera already initialized.")
            self.camera_initialized.emit(True)
            return

        try:
            self.status_updated.emit("Searching for Thorlabs camera...")
            # ThorCam() might block, which is why this is in a worker thread
            self.camera = ThorCam()
            self.status_updated.emit("Camera found. Initializing...")
            # Further setup might be needed here depending on camera model
            # For example: self.camera.set_exposure(10)
            self._is_initialized = True
            self.status_updated.emit("Camera initialized successfully.")
            self.camera_initialized.emit(True)
        except Exception as e:
            error_msg = f"Failed to initialize camera: {e}"
            self.error_occurred.emit(error_msg)
            self.camera = None
            self._is_initialized = False
            self.camera_initialized.emit(False)

    @pyqtSlot()
    def acquire_image(self):
        """
        Captures a single frame from the camera.
        """
        if not self._is_initialized or not self.camera:
            self.error_occurred.emit("Cannot acquire image. Camera not initialized.")
            return

        try:
            self.status_updated.emit("Acquiring image...")
            # The fetch_image() method should return a numpy array
            image_array = self.camera.fetch_image()
            if image_array is not None:
                self.status_updated.emit(f"Image acquired with shape: {image_array.shape}")
                self.image_acquired.emit(image_array)
            else:
                self.error_occurred.emit("Failed to acquire image (received None).")
        except Exception as e:
            self.error_occurred.emit(f"Error during image acquisition: {e}")

    @pyqtSlot()
    def cleanup_camera(self):
        """
        Releases the camera resources.
        """
        if self.camera:
            self.status_updated.emit("Releasing camera resources...")
            try:
                self.camera.close()
                self.camera = None
                self._is_initialized = False
                self.status_updated.emit("Camera resources released.")
            except Exception as e:
                self.error_occurred.emit(f"Error closing camera: {e}")

    @staticmethod
    def numpy_to_qimage(array):
        """
        Converts a NumPy array to a QImage.
        Assumes the input array is a 2D grayscale image (H, W).
        """
        if array is None:
            return QImage()
        
        height, width = array.shape
        bytes_per_line = width
        
        # The QImage constructor expects data to be contiguous.
        # The format depends on the numpy array's dtype.
        if array.dtype == np.uint8:
            return QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        elif array.dtype == np.uint16:
            # QImage doesn't have a 16-bit grayscale format directly.
            # We can normalize to 8-bit for display purposes.
            normalized_array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            return QImage(normalized_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        else:
            # Handle other types or raise an error
            print(f"Unsupported numpy dtype for QImage conversion: {array.dtype}")
            return QImage()


def run_camera_test():
    """
    A simple standalone test function to verify camera operation.
    This requires a QApplication instance to handle signals/slots.
    """
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    print("--- Camera Control Test ---")
    
    # We need a thread to run the manager
    camera_thread = QThread()
    camera_manager = CameraManager()
    camera_manager.moveToThread(camera_thread)

    # Connect signals to print statements
    camera_manager.status_updated.connect(lambda msg: print(f"[STATUS] {msg}"))
    camera_manager.error_occurred.connect(lambda err: print(f"[ERROR] {err}"))
    camera_manager.camera_initialized.connect(lambda success: print(f"[INIT] Success: {success}"))
    camera_manager.image_acquired.connect(lambda img: print(f"[IMAGE] Acquired image with shape: {img.shape}, dtype: {img.dtype}"))

    # Start the thread
    camera_thread.start()

    # --- Test Sequence ---
    print("\n1. Requesting camera initialization...")
    QThread.msleep(100) # Give thread time to start
    camera_manager.initialize_camera()
    QThread.msleep(3000) # Wait for initialization

    if camera_manager._is_initialized:
        print("\n2. Requesting image acquisition...")
        camera_manager.acquire_image()
        QThread.msleep(1000) # Wait for acquisition
    else:
        print("\nSkipping image acquisition due to initialization failure.")

    print("\n3. Requesting camera cleanup...")
    camera_manager.cleanup_camera()
    QThread.msleep(1000) # Wait for cleanup

    # --- Shutdown ---
    print("\n--- Test Finished ---")
    camera_thread.quit()
    camera_thread.wait(2000)
    app.quit()


if __name__ == '__main__':
    # This allows you to run this file directly to test camera functionality.
    run_camera_test()
