import numpy as np
import time
import sys
import os
import threading
import queue


try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
    from thorlabs_tsi_sdk.tl_color_enums import FORMAT
except ImportError:
    print("Error: 'thorlabs_tsi_sdk' library not found.")
    print("Please install it from the Thorlabs examples repository.")
    sys.exit(1)

for p in os.environ['PATH'].split(os.pathsep):
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except (FileNotFoundError, TypeError):
            pass # Ignore paths that don't exist or are not unicode



class CircularQueue(queue.Queue):
    """
    A thread-safe queue with a fixed size that discards the oldest item
    when it is full.

    This class inherits from queue.Queue and overrides the put() method
    to implement a circular buffer behavior. When a new item is put into a
    full queue, the oldest item is removed to make space.
    """
    def put(self, item, block=True, timeout=None):
        """
        Put an item into the queue.

        If the queue is full, it removes the oldest item and then adds the new
        item. This operation is thread-safe. The `block` and `timeout`
        arguments are ignored to maintain non-blocking behavior on a full queue
        but are kept for compatibility with the base class method signature.
        """
        with self.not_full:
            if self.maxsize > 0:
                if self._qsize() >= self.maxsize:
                    try:
                        self._get()
                    except queue.Empty:
                        pass
            
            self._put(item)
            self.not_empty.notify()

    def put_nowait(self, item):
        """
        A non-blocking variant of put().
        This is equivalent to calling put(item, block=False).
        """
        return self.put(item, block=False)



class _ImageAcquisitionThread(threading.Thread):
    """
    (Internal) A thread that continuously acquires images and places them into a queue.
    """
    def __init__(self, camera: TLCamera):
        super().__init__()
        self._camera = camera
        self._is_color = False
        # ... (The rest of this class is identical to the previous version)
        if self._camera.camera_sensor_type == SENSOR_TYPE.BAYER:
            self._is_color = True
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER, self._camera.color_filter_array_phase,
                self._camera.get_color_correction_matrix(), self._camera.get_default_white_balance_matrix(),
                self._camera.bit_depth)
        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = CircularQueue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self) -> queue.Queue:
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _process_frame(self, frame) -> np.ndarray:
        if self._is_color:
            color_image_data = self._mono_to_color_processor.transform_to_24(
                frame.image_buffer, self._camera.image_width_pixels, self._camera.image_height_pixels)
            return color_image_data.reshape(self._camera.image_height_pixels, self._camera.image_width_pixels, 3)
        else:
            scaled_image = frame.image_buffer >> (self._bit_depth - 8)
            return scaled_image.astype(np.uint8)

    def run(self):
        self._frame_cnt = 0
        print("Image acquisition thread started.")
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    image_np = self._process_frame(frame)
                    self._image_queue.put_nowait((image_np, self._frame_cnt))
                    self._frame_cnt += 1
            except queue.Full:
                pass
            except Exception as e:
                print(f"Error in acquisition thread: {e}")
                break
        print("Image acquisition thread has stopped.")
        if self._is_color:
            if self._mono_to_color_processor: self._mono_to_color_processor.dispose()
            if self._mono_to_color_sdk: self._mono_to_color_sdk.dispose()


# ------------------------------------------------------------------------------------
#  New Controller Class (Public Interface)
#  This is the new, simplified class you will use in your main script.
# ------------------------------------------------------------------------------------
class ThorlabsCameraController:
    """
    A high-level controller for a Thorlabs camera stream.

    This class handles all SDK initialization, camera connection, and thread
    management internally. Use it with a 'with' statement to ensure
    resources are properly released.
    """
    def __init__(self, camera_index: int = 0):
        """
        Initializes and starts the camera stream.
        :param camera_index: The index of the camera to connect to (e.g., 0 for the first camera).
        """
        self._sdk = None
        self._camera = None
        self._acquisition_thread = None
        self._image_queue = None
        
        try:
            print("Initializing Thorlabs SDK...")
            self._sdk = TLCameraSDK()
            camera_list = self._sdk.discover_available_cameras()
            if not camera_list:
                raise ConnectionError("No cameras detected.")
            if camera_index >= len(camera_list):
                raise IndexError(f"Camera index {camera_index} is out of range. "
                                 f"Found {len(camera_list)} cameras.")

            print(f"Opening camera at index {camera_index}...")
            self._camera = self._sdk.open_camera(camera_list[camera_index])
            
            # Start the background acquisition thread
            self._acquisition_thread = _ImageAcquisitionThread(self._camera)
            self._image_queue = self._acquisition_thread.get_output_queue()
            self._acquisition_thread.start()

            # Configure and start camera streaming
            self._camera.frames_per_trigger_zero_for_unlimited = 0
            self._camera.arm(2)
            self._camera.issue_software_trigger()
            print(f"Successfully started stream from {self._camera.name}.")

        except Exception as e:
            self.close()  # Cleanup on failure
            raise e

    def get_nowait(self) -> np.ndarray | None:
        """
        Gets the latest frame from the camera without blocking.
        :return: A NumPy array of the image, or None if no new frame is available.
        """
        try:
            return self._image_queue.get_nowait()
        except queue.Empty:
            return (None, None)

    def close(self):
        """Stops the stream and releases all resources."""
        print("Closing camera controller...")
        if self._acquisition_thread:
            self._acquisition_thread.stop()
            self._acquisition_thread.join()

        if self._camera:
            if self._camera.is_armed:
                self._camera.disarm()
            self._camera.dispose()

        if self._sdk:
            self._sdk.dispose()
        print("Resources released.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()