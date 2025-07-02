import sys
import os
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage
import time

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
except ImportError:
    print("Error: 'thorlabs_tsi_sdk' library not found.")
    print("Please install it from the Thorlabs examples repository.")
    sys.exit(1)

for p in os.environ['PATH'].split(os.pathsep):
    if os.path.isdir(p):
        os.add_dll_directory(p)

class CameraManager(QObject):
    """
    A QObject class to manage a Thorlabs camera in a separate thread using the official SDK package.
    """
    image_acquired = pyqtSignal(np.ndarray)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    camera_initialized = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.sdk = None
        self.camera = None
        self._is_initialized = False

    @pyqtSlot()
    def initialize_camera(self):
        if self._is_initialized:
            self.status_updated.emit("Camera already initialized.")
            self.camera_initialized.emit(True)
            return

        try:
            self.status_updated.emit("Initializing Thorlabs SDK...")
            self.sdk = TLCameraSDK()
            available_cameras = self.sdk.discover_available_cameras()
            if not available_cameras:
                raise Exception("No Thorlabs cameras found.")

            self.status_updated.emit(f"Found cameras: {available_cameras}")
            camera_sn = available_cameras[0]
            
            self.camera = self.sdk.open_camera(camera_sn)
            
            # Basic configuration
            self.camera.exposure_time_us = 10000  # 10 ms
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.image_poll_timeout_ms = 5000 # Increased timeout

            self._is_initialized = True
            self.status_updated.emit(f"Camera {camera_sn} initialized successfully.")
            self.camera_initialized.emit(True)

        except Exception as e:
            error_msg = f"Failed to initialize camera: {e}"
            self.error_occurred.emit(error_msg)
            self.cleanup_camera()
            self.camera_initialized.emit(False)

    @pyqtSlot()
    def snap_image(self):
        if not self._is_initialized or not self.camera:
            self.error_occurred.emit("Cannot acquire image. Camera not initialized.")
            return

        try:
            self.status_updated.emit("Acquiring image...")
            
            self.camera.arm(2) # Arm for 2 frames
            self.camera.issue_software_trigger()
            
            frame = self.camera.get_pending_frame_or_null()
            if frame is not None:
                image_array = np.copy(frame.image_buffer)
                self.image_acquired.emit(image_array)
                self.status_updated.emit(f"Image acquired with shape: {image_array.shape}")
            else:
                self.error_occurred.emit("Failed to acquire image (timeout).")

            self.camera.disarm()

        except Exception as e:
            self.error_occurred.emit(f"Error during image acquisition: {e}")
            if self.camera:
                self.camera.disarm()

    @pyqtSlot()
    def cleanup_camera(self):
        self.status_updated.emit("Releasing camera resources...")
        if self.camera:
            self.camera.dispose()
            self.camera = None
        
        if self.sdk:
            self.sdk.dispose()
            self.sdk = None
        
        self._is_initialized = False
        self.status_updated.emit("Camera resources released.")

    @staticmethod
    def numpy_to_qimage(array):
        if array is None:
            return QImage()
        
        if array.dtype != np.uint8:
             array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

        height, width = array.shape
        bytes_per_line = width
        return QImage(array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)


def run_camera_test():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication

    app = QApplication(sys.argv)
    print("--- Camera Control Test (Official SDK Package) ---")
    
    camera_thread = QThread()
    camera_manager = CameraManager()
    camera_manager.moveToThread(camera_thread)

    camera_manager.status_updated.connect(lambda msg: print(f"[STATUS] {msg}"))
    camera_manager.error_occurred.connect(lambda err: print(f"[ERROR] {err}"))
    camera_manager.image_acquired.connect(lambda img: print(f"[IMAGE] Acquired image with shape: {img.shape}, dtype: {img.dtype}"))
    
    is_initialized = False
    def on_initialized(success):
        nonlocal is_initialized
        is_initialized = success
        print(f"[INIT] Success: {success}")

    camera_manager.camera_initialized.connect(on_initialized)
    camera_thread.start()

    print("\n1. Requesting camera initialization...")
    camera_manager.initialize_camera()
    
    for _ in range(10):
        if is_initialized:
            break
        QThread.msleep(500)
    
    if is_initialized:
        print("\n2. Requesting image snap...")
        camera_manager.snap_image()
        QThread.msleep(1000)
    else:
        print("\nSkipping image snap due to initialization failure.")

    print("\n3. Requesting camera cleanup...")
    camera_manager.cleanup_camera()
    QThread.msleep(1000)

    print("\n--- Test Finished ---")
    camera_thread.quit()
    camera_thread.wait(2000)
    QCoreApplication.instance().quit()

if __name__ == '__main__':
    run_camera_test()
