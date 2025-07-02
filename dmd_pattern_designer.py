import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem, QCheckBox,
    QMessageBox, QSizePolicy, QSpacerItem, QFrame, QStatusBar
)
from PyQt6.QtCore import Qt, QPointF, QRectF, QThread, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QTransform, QFontMetrics, QPixmap
import math

# Try to import the Camera and DMD control modules
try:
    from camera_control import CameraManager
except ImportError:
    QMessageBox.critical(None, "Import Error", "Could not import 'CameraManager' from 'camera_control.py'.")
    sys.exit(1)
try:
    from pattern_on_the_fly import PatternOnTheFly
except ImportError:
    # This message box might not be ideal if running headlessly, but good for GUI app
    app_temp = QApplication.instance() # Check if app exists
    if not app_temp:
        app_temp = QApplication(sys.argv) # Create temp for message box
    
    QMessageBox.critical(None, "Import Error",
                         "Could not import 'PatternOnTheFly' from 'pattern_on_the_fly.py'.\n"
                         "Please ensure the file is in the same directory or in the Python path, "
                         "and that it's a valid module.")
    if not QApplication.instance(): # if we created app_temp, and it's the only one
        app_temp.quit()
    sys.exit(1)


DMD_WIDTH = 1920
DMD_HEIGHT = 1080
DEFAULT_REGION_SIZE = 50
DEFAULT_GRID_SPACING = 50
DEFAULT_ON_DURATION_MS = 100.0
DEFAULT_OFF_DURATION_MS = 100.0
DEFAULT_TOTAL_DURATION_MS = 1000.0

class PatternCanvas(QGraphicsView):
    """
    A QGraphicsView widget for interactively designing spatial patterns.
    Displays a representation of the DMD surface where users can define active regions.
    """
    pattern_updated_signal = pyqtSignal(np.ndarray)
    scale_changed_signal = pyqtSignal(float) # To update zoom slider

    MIN_ZOOM_SCALE = 0.1
    MAX_ZOOM_SCALE = 10.0 # User feedback indicated previous version ran, likely with 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, DMD_WIDTH, DMD_HEIGHT)
        self.setScene(self.scene)

        self.numpy_pattern = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
        self.active_regions_items = []
        self.background_pixmap = None

        # Shape drawing parameters
        self.current_shape_mode = "Square" # Default shape
        self.current_shape_size = DEFAULT_REGION_SIZE # Side for square, diameter for circle

        self.show_grid = False
        self.base_grid_spacing = DEFAULT_GRID_SPACING
        self.grid_pen = QPen(QColor(Qt.GlobalColor.darkGray), 0.5, Qt.PenStyle.SolidLine)
        self.grid_label_pen = QPen(QColor(Qt.GlobalColor.lightGray), 1) # Pen for grid labels
        self.dmd_outline_pen = QPen(QColor(Qt.GlobalColor.red), 2, Qt.PenStyle.SolidLine)

        self._current_scale_factor = 1.0
        self._is_fitting_in_view = False # Flag to prevent recursion during initial fitInView

        self._setup_ui()
        self.clear_pattern()

    def _setup_ui(self):
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.scene.setBackgroundBrush(Qt.GlobalColor.black)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setMinimumSize(400, 300)

    def set_shape_params(self, mode, size):
        self.current_shape_mode = mode
        self.current_shape_size = max(1, size)
        # No direct visual update needed here, happens on next click or pattern modification

    def toggle_grid(self, show):
        self.show_grid = show
        self.viewport().update() # Request full viewport repaint

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.sceneRect().contains(self.mapToScene(event.pos())):
            scene_pos = self.mapToScene(event.pos())
            self._add_active_region(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def _add_active_region(self, center_x, center_y):
        size = self.current_shape_size
        item_to_add = None

        if self.current_shape_mode == "Square":
            half_size = size / 2.0
            item_x_float = center_x - half_size
            item_y_float = center_y - half_size
            
            item_x = int(round(item_x_float))
            item_y = int(round(item_y_float))
            item_w = int(round(size))
            item_h = int(round(size))

            item_to_add = QGraphicsRectItem(item_x, item_y, item_w, item_h)
            
            np_x_start = max(0, item_x)
            np_y_start = max(0, item_y)
            np_x_end = min(DMD_WIDTH, item_x + item_w)
            np_y_end = min(DMD_HEIGHT, item_y + item_h)

            if np_x_start < np_x_end and np_y_start < np_y_end:
                self.numpy_pattern[np_y_start:np_y_end, np_x_start:np_x_end] = 1

        elif self.current_shape_mode == "Circle":
            # 1. Snap parameters to pixel grid
            pixel_diameter = max(1, int(round(float(size))))
            pixel_radius = pixel_diameter / 2.0
            
            # Center of the circle, aligned to pixel coordinates (integer)
            # This means the conceptual center of the circle is at the center of a pixel, or at an intersection of 4 pixels.
            # Let's use the center of the pixel closest to the click.
            pixel_center_x = round(center_x) 
            pixel_center_y = round(center_y) 

            # No single QGraphicsEllipseItem anymore for visual representation.
            # We will add individual 1x1 QGraphicsRectItems for each pixel.
            item_to_add = None # Will not be used in the same way for circles.
            
            # 2. Rasterize into NumPy array using pixel centers AND add 1x1 pixel rects
            np_center_x_coord = round(center_x) 
            np_center_y_coord = round(center_y)
            pixel_radius_sq = pixel_radius * pixel_radius

            # Determine integer pixel indices for iteration range
            # Bounding box for iteration:
            min_iter_x = max(0, int(math.floor(np_center_x_coord - pixel_radius)))
            max_iter_x = min(DMD_WIDTH, int(math.ceil(np_center_x_coord + pixel_radius)))
            min_iter_y = max(0, int(math.floor(np_center_y_coord - pixel_radius)))
            max_iter_y = min(DMD_HEIGHT, int(math.ceil(np_center_y_coord + pixel_radius)))

            for r_idx in range(min_iter_y, max_iter_y):  # Iterate through pixel row indices
                for c_idx in range(min_iter_x, max_iter_x): # Iterate through pixel column indices
                    # Calculate distance from the *center* of the current pixel (c_idx + 0.5, r_idx + 0.5)
                    # to the *pixel-aligned center* of the circle.
                    dist_sq = ((c_idx + 0.5) - np_center_x_coord)**2 + \
                              ((r_idx + 0.5) - np_center_y_coord)**2
                    
                    if dist_sq <= pixel_radius_sq:
                        # This pixel's center is within the circle
                        if 0 <= r_idx < DMD_HEIGHT and 0 <= c_idx < DMD_WIDTH: # Redundant check if iter range is correct, but safe
                            self.numpy_pattern[r_idx, c_idx] = 1
                            
                            # Add a 1x1 QGraphicsRectItem for this pixel
                            pixel_rect = QGraphicsRectItem(float(c_idx), float(r_idx), 1.0, 1.0)
                            pixel_rect.setBrush(Qt.GlobalColor.white)
                            pixel_rect.setPen(QPen(Qt.PenStyle.NoPen)) # No border for individual pixels
                            self.scene.addItem(pixel_rect)
                            self.active_regions_items.append(pixel_rect)
            
            # Signal update once after all pixels for the circle are processed
            self.pattern_updated_signal.emit(self.numpy_pattern)
            # No single 'item_to_add' for circles in this pixelated approach
            return # Exit early as items are added directly

        # Common logic for Square (and if Circle had a single item, which it doesn't anymore)
        if item_to_add: # This will only be true for Squares now
            item_to_add.setBrush(Qt.GlobalColor.white)
            item_to_add.setPen(QPen(Qt.PenStyle.NoPen))
            self.scene.addItem(item_to_add)
            self.active_regions_items.append(item_to_add)
            self.pattern_updated_signal.emit(self.numpy_pattern) # Square already emitted, but safe

    def _update_transform(self):
        if self._is_fitting_in_view: # Avoid transform changes during fitInView
            return
        
        transform = QTransform()
        transform.scale(self._current_scale_factor, self._current_scale_factor)
        self.setTransform(transform)
        self.scale_changed_signal.emit(self._current_scale_factor)
        self.viewport().update() # Ensure grid and foreground redraw

    def set_absolute_zoom(self, scale_factor):
        self._current_scale_factor = max(self.MIN_ZOOM_SCALE, min(scale_factor, self.MAX_ZOOM_SCALE))
        self._update_transform()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier: # Or another modifier if preferred
            zoom_factor_delta = 1.15
            if event.angleDelta().y() > 0: # Zoom in
                new_scale = self._current_scale_factor * zoom_factor_delta
            else: # Zoom out
                new_scale = self._current_scale_factor / zoom_factor_delta
            self.set_absolute_zoom(new_scale)
        else:
            super().wheelEvent(event) # Default scroll behavior for panning if Ctrl not held

    def _calculate_adaptive_grid_spacing(self):
        # Aim for grid lines to be roughly 50-100 screen pixels apart
        target_screen_pixels = 75 
        
        # Current horizontal scale factor from the view's transform
        # m11 is the horizontal scaling component
        view_scale = self.transform().m11()
        if view_scale == 0: view_scale = 1.0 # Avoid division by zero if transform is weird

        # How many scene units correspond to one screen pixel
        scene_units_per_screen_pixel = 1.0 / view_scale

        # Ideal grid spacing in scene units to achieve target_screen_pixels density
        ideal_scene_spacing = target_screen_pixels * scene_units_per_screen_pixel

        if ideal_scene_spacing <= 0: # Should not happen with positive view_scale
            return self.base_grid_spacing

        # Snap to "nice" values (e.g., multiples of 1, 2, 5, 10, 20, 25, 50, 100...)
        # This is a simplified snapping logic. More sophisticated might be needed.
        # We want the dynamic spacing to be a multiple of some base unit, or derived from base_grid_spacing
        
        # Find a power of 10 close to ideal_scene_spacing
        power = math.pow(10, math.floor(math.log10(ideal_scene_spacing)))
        
        # Check multiples 1, 2, 5 of that power
        multiples = [1 * power, 2 * power, 5 * power, 10 * power]
        
        best_spacing = multiples[-1] # Default to largest
        min_diff = float('inf')

        for s in multiples:
            if s < 0.1 : continue # Avoid too small grid spacing that might cause freeze
            diff = abs(s - ideal_scene_spacing)
            if diff < min_diff:
                min_diff = diff
                best_spacing = s
        
        # Ensure minimum spacing to prevent freezing
        return max(1.0, best_spacing)


    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)

        if self.background_pixmap:
            painter.drawPixmap(rect, self.background_pixmap, rect)
        
        if self.show_grid and self.base_grid_spacing > 0:
            current_adaptive_spacing = self._calculate_adaptive_grid_spacing()
            if current_adaptive_spacing < 1: # Safety for extremely small spacing
                return

            painter.setPen(self.grid_pen)
            
            visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()

            left = visible_scene_rect.left()
            top = visible_scene_rect.top()
            right = visible_scene_rect.right()
            bottom = visible_scene_rect.bottom()

            # Draw vertical lines
            # Align to 0,0 of the scene for consistency
            x_start_grid = math.floor(left / current_adaptive_spacing) * current_adaptive_spacing
            for x in np.arange(x_start_grid, right, current_adaptive_spacing):
                if -DMD_WIDTH*2 < x < DMD_WIDTH * 2: # Generous bounds to avoid missing lines at edges, but prevent excessive drawing
                    painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            
            # Draw horizontal lines
            y_start_grid = math.floor(top / current_adaptive_spacing) * current_adaptive_spacing
            for y in np.arange(y_start_grid, bottom, current_adaptive_spacing):
                 if -DMD_HEIGHT*2 < y < DMD_HEIGHT*2:
                    painter.drawLine(QPointF(left, y), QPointF(right, y))

    def drawForeground(self, painter, rect):
        """Draws elements on top of scene items, like the DMD outline and coordinate labels."""
        super().drawForeground(painter, rect)
        
        # Draw DMD Outline
        painter.setPen(self.dmd_outline_pen)
        dmd_boundary = QRectF(0, 0, DMD_WIDTH, DMD_HEIGHT)
        painter.drawRect(dmd_boundary)

        # Draw Grid Labels if grid is shown and spacing is reasonable
        if self.show_grid and self.base_grid_spacing > 0:
            current_adaptive_spacing = self._calculate_adaptive_grid_spacing()
            # Only draw labels if adaptive spacing is large enough to avoid clutter
            # And if current scale is not too small (e.g. > 0.2x)
            if current_adaptive_spacing >= 20 and self._current_scale_factor > 0.15: # Thresholds can be tuned
                self._draw_grid_labels(painter, current_adaptive_spacing)

    def _draw_grid_labels(self, painter, spacing):
        painter.setPen(self.grid_label_pen)
        font = painter.font()
        font.setPointSize(8) # Fixed small font size for labels
        painter.setFont(font)

        visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        
        # Label offsets from the lines - these are in scene coordinates now if font is fixed
        label_offset_x = 8 
        label_offset_y = 5 
        # Adjust Y offset for X-labels to be further down
        x_label_y_offset_multiplier = 5

        # Vertical lines (X-coordinates)
        x_start_grid = math.floor(visible_scene_rect.left() / spacing) * spacing
        for x_coord in np.arange(x_start_grid, visible_scene_rect.right(), spacing):
            if 0 <= x_coord <= DMD_WIDTH:
                text_point = QPointF(x_coord + label_offset_x, visible_scene_rect.top() + label_offset_y * x_label_y_offset_multiplier)
                
                # Simplified check: if the line itself is visible, attempt to draw label near top-right of it.
                # More robust culling would check if text_point (after rotation) is in viewport.
                # if visible_scene_rect.contains(text_point): # This check might be too restrictive for rotated text
                painter.save()
                painter.translate(text_point)
                painter.rotate(-90) 
                painter.drawText(QPointF(0,0), f"{int(x_coord)}") # Draw at new origin (0,0)
                painter.restore()

        # Horizontal lines (Y-coordinates)
        y_start_grid = math.floor(visible_scene_rect.top() / spacing) * spacing
        for y_coord in np.arange(y_start_grid, visible_scene_rect.bottom(), spacing):
            if 0 <= y_coord <= DMD_HEIGHT:
                text_point = QPointF(visible_scene_rect.left() + label_offset_x, y_coord - label_offset_y)
                # if visible_scene_rect.contains(text_point): # Similar to above, might be too restrictive
                painter.drawText(text_point, f"{int(y_coord)}")


    def clear_pattern(self):
        for item in self.active_regions_items:
            if item.scene() == self.scene: # Ensure item is still in scene
                 self.scene.removeItem(item)
        self.active_regions_items.clear()
        self.numpy_pattern.fill(0)
        self.pattern_updated_signal.emit(self.numpy_pattern)
        self.scene.update()

    def set_background_image(self, pixmap):
        self.background_pixmap = pixmap
        self.viewport().update()

    def get_pattern_array(self):
        return self.numpy_pattern.copy()

    def showEvent(self, event):
        # Fit the entire scene in view when the widget is first shown
        # This might trigger wheelEvent if not handled, or transform updates.
        if not self._is_fitting_in_view: # Check flag
            self._is_fitting_in_view = True
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            # After fitInView, the transform is set. We need to update our _current_scale_factor
            # Assuming fitInView results in a QTransform that is purely scaling and translation
            current_transform = self.transform()
            self._current_scale_factor = current_transform.m11() # Horizontal scale
            # self._current_scale_factor could also be m22() if aspect ratio is preserved.
            self.scale_changed_signal.emit(self._current_scale_factor)
            self._is_fitting_in_view = False
        super().showEvent(event)


class DMDWorker(QObject):
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    operation_finished = pyqtSignal(bool) # bool indicates success

    def __init__(self):
        super().__init__()
        self.dmd_instance = None
        self._is_dmd_initialized = False

    @pyqtSlot()
    def initialize_dmd(self):
        if self._is_dmd_initialized and self.dmd_instance:
            self.status_updated.emit("DMD already initialized.")
            self.operation_finished.emit(True)
            return
        try:
            self.status_updated.emit("Initializing DMD...")
            # Using w, h and test=True as per typical usage and previous feedback context
            self.dmd_instance = PatternOnTheFly(w=DMD_WIDTH, h=DMD_HEIGHT, test=False)
            self._is_dmd_initialized = True
            self.status_updated.emit("DMD initialized successfully.")
            self.operation_finished.emit(True)
        except Exception as e:
            self.error_occurred.emit(f"DMD Initialization Error: {e}")
            self.dmd_instance = None
            self._is_dmd_initialized = False
            self.operation_finished.emit(False)

    @pyqtSlot(np.ndarray, float, float, float)
    def send_sequence_to_dmd(self, pattern_array, on_duration_ms, off_duration_ms, total_sequence_duration_ms):
        if not self._is_dmd_initialized or not self.dmd_instance:
            self.error_occurred.emit("DMD not initialized. Cannot send sequence.")
            self.operation_finished.emit(False)
            return

        try:
            self.status_updated.emit("Processing sequence...")

            if on_duration_ms <= 0:
                self.error_occurred.emit("On Duration must be positive.")
                self.operation_finished.emit(False); return
            if total_sequence_duration_ms <= 0:
                self.error_occurred.emit("Total Sequence Duration must be positive.")
                self.operation_finished.emit(False); return
            if off_duration_ms < 0:
                self.error_occurred.emit("Off Duration cannot be negative.")
                self.operation_finished.emit(False); return

            exposure_us = int(on_duration_ms * 1000)
            darktime_us = int(off_duration_ms * 1000)
            
            num_repeats = 0
            cycle_duration_ms = on_duration_ms + off_duration_ms

            if cycle_duration_ms > 0:
                num_repeats = int(total_sequence_duration_ms // cycle_duration_ms)
                if num_repeats == 0 and total_sequence_duration_ms >= on_duration_ms:
                    num_repeats = 1 
            elif on_duration_ms > 0: # Static display
                if total_sequence_duration_ms >= on_duration_ms:
                    num_repeats = 1
                    darktime_us = 0 
            
            if num_repeats <= 0:
                self.error_occurred.emit("Calculated nRepeat is <= 0 or total duration too short. No sequence sent.")
                self.operation_finished.emit(False)
                return

            self.status_updated.emit(f"Sending: On={on_duration_ms}ms, Off={off_duration_ms}ms, Total={total_sequence_duration_ms}ms, Repeats={num_repeats}")
            
            if not pattern_array.flags['C_CONTIGUOUS']:
                pattern_array = np.ascontiguousarray(pattern_array, dtype=np.uint8)
            if pattern_array.dtype != np.uint8:
                 pattern_array = pattern_array.astype(np.uint8)

            self.dmd_instance.DefinePattern(0, exposure=exposure_us, darktime=darktime_us, data=pattern_array)
            self.dmd_instance.SendImageSequence(nPattern=1, nRepeat=num_repeats)
            self.dmd_instance.StartRunning()
            
            self.status_updated.emit("Sequence sent and started on DMD.")
            self.operation_finished.emit(True)

        except Exception as e:
            self.error_occurred.emit(f"DMD Operation Error: {e}")
            self.operation_finished.emit(False)

    @pyqtSlot()
    def stop_dmd_sequence(self):
        if not self._is_dmd_initialized or not self.dmd_instance:
            self.error_occurred.emit("DMD not initialized. Cannot stop.")
            self.operation_finished.emit(False)
            return
        try:
            self.status_updated.emit("Stopping DMD sequence...")
            self.dmd_instance.StopRunning()
            self.status_updated.emit("DMD sequence stopped.")
            self.operation_finished.emit(True)
        except Exception as e:
            self.error_occurred.emit(f"DMD Stop Error: {e}")
            self.operation_finished.emit(False)
            
    @pyqtSlot()
    def cleanup_dmd(self):
        if self.dmd_instance:
            try:
                self.status_updated.emit("Cleaning up DMD resources...")
                # Check if 'is_running' attribute exists (for mock or real API)
                is_running = False
                if hasattr(self.dmd_instance, 'is_running'):
                    is_running = self.dmd_instance.is_running
                elif hasattr(self.dmd_instance, 'get_status'): # Example alternative
                    status = self.dmd_instance.get_status()
                    is_running = status.get('running', False)

                if is_running:
                    self.dmd_instance.StopRunning()
                
                if hasattr(self.dmd_instance, 'close'): # Explicit close method
                    self.dmd_instance.close()
                elif hasattr(self.dmd_instance, 'release'): # Explicit release method
                    self.dmd_instance.release()
                
                del self.dmd_instance 
                self.dmd_instance = None
                self._is_dmd_initialized = False
                self.status_updated.emit("DMD resources cleaned up.")
            except Exception as e:
                self.error_occurred.emit(f"DMD Cleanup Error: {e}")


class MainWindow(QMainWindow):
    # DMD Signals
    request_initialize_dmd = pyqtSignal()
    request_send_sequence = pyqtSignal(np.ndarray, float, float, float)
    request_stop_dmd = pyqtSignal()
    request_cleanup_dmd = pyqtSignal()

    # Camera Signals
    request_initialize_camera = pyqtSignal()
    request_snap_image = pyqtSignal()
    request_cleanup_camera = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DMD Pattern Stimulator")
        self.setGeometry(100, 100, 1200, 800)

        self._setup_ui()
        self._setup_dmd_thread()
        self._setup_camera_thread()
        self._connect_signals()

        self.request_initialize_dmd.emit()
        self.request_initialize_camera.emit()
        self._set_buttons_enabled_state(dmd_ready=False, sequence_running=False)

    def _set_buttons_enabled_state(self, dmd_ready, sequence_running):
        self.send_button.setEnabled(dmd_ready and not sequence_running)
        self.stop_button.setEnabled(dmd_ready) # Enabled if DMD is ready, regardless of sequence_running
        # Other controls can be disabled during sequence running if needed
        self.clear_pattern_button.setEnabled(not sequence_running)
        self.shape_type_combo.setEnabled(not sequence_running)
        self.shape_size_spinbox.setEnabled(not sequence_running)
        self.show_grid_checkbox.setEnabled(not sequence_running)
        self.on_duration_spinbox.setEnabled(not sequence_running)
        self.off_duration_spinbox.setEnabled(not sequence_running)
        self.total_duration_spinbox.setEnabled(not sequence_running)
        self.pattern_canvas.setEnabled(not sequence_running)


    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Instantiate PatternCanvas first as other UI elements might depend on it
        self.pattern_canvas = PatternCanvas()

        # Left Panel: Controls
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_panel.setMaximumWidth(400)

        # Spatial Pattern Editor Controls
        spatial_group = QFrame()
        spatial_group.setFrameShape(QFrame.Shape.StyledPanel)
        spatial_layout = QGridLayout()
        spatial_group.setLayout(spatial_layout)
        
        row = 0
        spatial_layout.addWidget(QLabel("<b>Spatial Pattern Editor</b>"), row, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter); row += 1
        
        spatial_layout.addWidget(QLabel("Shape Type:"), row, 0)
        self.shape_type_combo = QComboBox()
        self.shape_type_combo.addItems(["Square", "Circle"])
        spatial_layout.addWidget(self.shape_type_combo, row, 1); row += 1

        self.shape_size_label = QLabel("Side Length (px):") # Dynamic label
        spatial_layout.addWidget(self.shape_size_label, row, 0)
        self.shape_size_spinbox = QSpinBox() # Replaces region_width_spinbox
        self.shape_size_spinbox.setRange(1, max(DMD_WIDTH, DMD_HEIGHT))
        self.shape_size_spinbox.setValue(DEFAULT_REGION_SIZE)
        spatial_layout.addWidget(self.shape_size_spinbox, row, 1); row += 1
        
        # Region Height spinbox and its label are removed.

        self.show_grid_checkbox = QCheckBox("Show Grid")
        self.show_grid_checkbox.setChecked(False)
        spatial_layout.addWidget(self.show_grid_checkbox, row, 0, 1, 2); row += 1

        self.clear_pattern_button = QPushButton("Clear Current Pattern")
        spatial_layout.addWidget(self.clear_pattern_button, row, 0, 1, 2); row += 1
        
        controls_layout.addWidget(spatial_group)

        # Zoom controls moved outside and below spatial_group
        zoom_control_group = QFrame() # Optional: group them visually
        zoom_control_group.setFrameShape(QFrame.Shape.StyledPanel)
        zoom_layout = QVBoxLayout(zoom_control_group)

        self.zoom_slider_label = QLabel("Zoom Level:") # Keep label separate for clarity
        zoom_layout.addWidget(self.zoom_slider_label)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(0, 100) 
        initial_slider_val = int( ( (1.0 - PatternCanvas.MIN_ZOOM_SCALE) / \
                                  (PatternCanvas.MAX_ZOOM_SCALE - PatternCanvas.MIN_ZOOM_SCALE) ) * 100 )
        self.zoom_slider.setValue(initial_slider_val if 0 <= initial_slider_val <= 100 else 50)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel(f"Zoom: {self.pattern_canvas._current_scale_factor:.2f}x")
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_layout.addWidget(self.zoom_label)
        
        controls_layout.addWidget(zoom_control_group)


        # Temporal Sequence Controls
        temporal_group = QFrame()
        temporal_group.setFrameShape(QFrame.Shape.StyledPanel)
        temporal_layout = QGridLayout()
        temporal_group.setLayout(temporal_layout)

        temporal_layout.addWidget(QLabel("<b>Temporal Sequence Controls</b>"), 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        temporal_layout.addWidget(QLabel("On Duration (ms):"), 1, 0)
        self.on_duration_spinbox = QDoubleSpinBox()
        self.on_duration_spinbox.setRange(0.1, 600000.0) # Max 10 mins
        self.on_duration_spinbox.setValue(DEFAULT_ON_DURATION_MS)
        self.on_duration_spinbox.setDecimals(1)
        temporal_layout.addWidget(self.on_duration_spinbox, 1, 1)

        temporal_layout.addWidget(QLabel("Off Duration (ms):"), 2, 0)
        self.off_duration_spinbox = QDoubleSpinBox()
        self.off_duration_spinbox.setRange(0.0, 600000.0)
        self.off_duration_spinbox.setValue(DEFAULT_OFF_DURATION_MS)
        self.off_duration_spinbox.setDecimals(1)
        temporal_layout.addWidget(self.off_duration_spinbox, 2, 1)

        temporal_layout.addWidget(QLabel("Total Sequence (ms):"), 3, 0)
        self.total_duration_spinbox = QDoubleSpinBox()
        self.total_duration_spinbox.setRange(0.1, 3600000.0) # Max 1 hour
        self.total_duration_spinbox.setValue(DEFAULT_TOTAL_DURATION_MS)
        self.total_duration_spinbox.setDecimals(1)
        temporal_layout.addWidget(self.total_duration_spinbox, 3, 1)
        controls_layout.addWidget(temporal_group)
        controls_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Camera Control Panel
        camera_control_group = QFrame()
        camera_control_group.setFrameShape(QFrame.Shape.StyledPanel)
        camera_control_layout = QVBoxLayout(camera_control_group)
        camera_control_layout.addWidget(QLabel("<b>Camera Control</b>"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.snap_button = QPushButton("Acquire Background Image")
        camera_control_layout.addWidget(self.snap_button)
        self.clear_background_button = QPushButton("Clear Background Image")
        camera_control_layout.addWidget(self.clear_background_button)
        controls_layout.addWidget(camera_control_group)

        # DMD Control Panel
        dmd_control_group = QFrame()
        dmd_control_group.setFrameShape(QFrame.Shape.StyledPanel)
        dmd_control_layout = QVBoxLayout() # Simpler layout for these buttons
        dmd_control_group.setLayout(dmd_control_layout)

        dmd_control_layout.addWidget(QLabel("<b>DMD Control Panel</b>"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.send_button = QPushButton("Send Sequence to DMD")
        self.send_button.setStyleSheet("background-color: lightgreen;")
        dmd_control_layout.addWidget(self.send_button)

        self.stop_button = QPushButton("Stop DMD Sequence")
        self.stop_button.setStyleSheet("background-color: salmon;")
        dmd_control_layout.addWidget(self.stop_button)
        controls_layout.addWidget(dmd_control_group)
        
        controls_layout.addStretch(1) # Pushes everything up
        
        main_layout.addWidget(controls_panel)
        main_layout.addWidget(self.pattern_canvas, 1) # Canvas takes more space

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Application started. Initializing DMD...")

    def _setup_dmd_thread(self):
        self.dmd_thread = QThread()
        self.dmd_worker = DMDWorker()
        self.dmd_worker.moveToThread(self.dmd_thread)

        # Connect worker signals to main thread slots
        self.dmd_worker.status_updated.connect(self.update_status_bar)
        self.dmd_worker.error_occurred.connect(self.show_error_message)
        self.dmd_worker.operation_finished.connect(self.on_dmd_operation_finished)

        # Connect main thread signals to worker slots
        self.request_initialize_dmd.connect(self.dmd_worker.initialize_dmd)
        self.request_send_sequence.connect(self.dmd_worker.send_sequence_to_dmd)
        self.request_stop_dmd.connect(self.dmd_worker.stop_dmd_sequence)
        self.request_cleanup_dmd.connect(self.dmd_worker.cleanup_dmd)
        
        self.dmd_thread.start()

    def _setup_camera_thread(self):
        self.camera_thread = QThread()
        self.camera_worker = CameraManager()
        self.camera_worker.moveToThread(self.camera_thread)

        # Connect worker signals to main thread slots
        self.camera_worker.status_updated.connect(self.update_status_bar)
        self.camera_worker.error_occurred.connect(self.show_error_message)
        self.camera_worker.image_acquired.connect(self.on_background_image_acquired)
        self.camera_worker.camera_initialized.connect(lambda ok: self.snap_button.setEnabled(ok))

        # Connect main thread signals to worker slots
        self.request_initialize_camera.connect(self.camera_worker.initialize_camera)
        self.request_snap_image.connect(self.camera_worker.snap_image)
        self.request_cleanup_camera.connect(self.camera_worker.cleanup_camera)
        
        self.camera_thread.start()

    def _connect_signals(self):
        # Spatial controls
        self.shape_type_combo.currentTextChanged.connect(self._on_shape_mode_changed)
        self.shape_size_spinbox.valueChanged.connect(self._on_shape_size_changed)
        
        self.show_grid_checkbox.toggled.connect(self.pattern_canvas.toggle_grid)
        self.clear_pattern_button.clicked.connect(self.pattern_canvas.clear_pattern)
        self.pattern_canvas.pattern_updated_signal.connect(self.on_pattern_updated)
        
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        self.pattern_canvas.scale_changed_signal.connect(self._on_canvas_scale_changed)

        # Initialize canvas shape params based on default UI
        self._on_shape_mode_changed(self.shape_type_combo.currentText()) # Call after pattern_canvas is available


        # DMD controls
        self.send_button.clicked.connect(self.on_send_sequence_clicked)
        self.stop_button.clicked.connect(self.on_stop_dmd_clicked)

        # Camera controls
        self.snap_button.clicked.connect(self.request_snap_image)
        self.clear_background_button.clicked.connect(lambda: self.pattern_canvas.set_background_image(None))

    @pyqtSlot(np.ndarray)
    def on_background_image_acquired(self, image_array):
        q_image = self.camera_worker.numpy_to_qimage(image_array)
        pixmap = QPixmap.fromImage(q_image)
        self.pattern_canvas.set_background_image(pixmap)
        self.update_status_bar("Background image updated.")

    def on_pattern_updated(self, pattern_array):
        # Could update some stats here if needed, e.g., number of active pixels
        active_pixels = np.sum(pattern_array)
        self.update_status_bar(f"Pattern updated. Active pixels: {active_pixels}")

    def on_send_sequence_clicked(self):
        pattern = self.pattern_canvas.get_pattern_array()
        if np.sum(pattern) == 0:
            QMessageBox.warning(self, "Empty Pattern", "The current pattern is empty. Add some active regions before sending.")
            return

        on_ms = self.on_duration_spinbox.value()
        off_ms = self.off_duration_spinbox.value()
        total_ms = self.total_duration_spinbox.value()

        if on_ms <= 0:
            QMessageBox.warning(self, "Invalid Input", "On Duration must be greater than 0 ms.")
            return
        if total_ms <= 0:
            QMessageBox.warning(self, "Invalid Input", "Total Sequence Duration must be greater than 0 ms.")
            return
        
        self._set_buttons_enabled_state(dmd_ready=True, sequence_running=True) # Assume DMD is ready
        self.status_bar.showMessage("Sending sequence to DMD...")
        self.request_send_sequence.emit(pattern, on_ms, off_ms, total_ms)

    def on_stop_dmd_clicked(self):
        self._set_buttons_enabled_state(dmd_ready=True, sequence_running=True) # Keep disabled until op finishes
        self.status_bar.showMessage("Stopping DMD sequence...")
        self.request_stop_dmd.emit()

    @pyqtSlot(str)
    def update_status_bar(self, message):
        self.status_bar.showMessage(message)

    @pyqtSlot(str)
    def show_error_message(self, message):
        QMessageBox.critical(self, "DMD Error", message)
        self.status_bar.showMessage(f"Error: {message}", 5000) # Show error in status bar for 5s

    @pyqtSlot(bool)
    def on_dmd_operation_finished(self, success):
        # This slot is crucial for re-enabling UI elements
        # Determine if a sequence is (still) considered running.
        # This might need more sophisticated state management if StartRunning is non-blocking
        # and we don't get an explicit "sequence_finished" signal from the hardware.
        # For now, assume 'send' starts it, 'stop' or error stops it.
        
        # A simple check: if the last operation was 'send' and it succeeded, assume running.
        # If 'stop' or an error, assume not running.
        # This logic needs refinement based on actual DMD behavior.
        # For this example, let's assume DMD is ready, and sequence is NOT running after any op.
        # This means user has to press Send again.
        # A better model would be:
        #   - DMD init success -> dmd_ready = true
        #   - Send success -> sequence_running = true
        #   - Stop success / Send error / Stop error -> sequence_running = false
        
        # Simplified: if DMD worker is initialized, it's ready.
        dmd_is_ready = self.dmd_worker._is_dmd_initialized
        
        # Check if the operation was a successful start
        # This is tricky without knowing which operation just finished.
        # Let's assume for now that after any operation, the sequence is NOT running
        # unless explicitly started.
        # A more robust way: DMDWorker could emit a signal sequence_started / sequence_stopped.
        
        # For now, let's re-enable based on dmd_is_ready and assume sequence is not running
        # until Send is pressed again.
        self._set_buttons_enabled_state(dmd_ready=dmd_is_ready, sequence_running=False)
        if dmd_is_ready and success:
             self.status_bar.showMessage("DMD operation complete.", 3000)
        elif not dmd_is_ready:
             self.status_bar.showMessage("DMD not ready.", 3000)


    def closeEvent(self, event):
        self.status_bar.showMessage("Shutting down...")
        
        # Cleanup Camera Thread
        if self.camera_thread.isRunning():
            self.request_cleanup_camera.emit()
            self.camera_thread.quit()
            if not self.camera_thread.wait(3000):
                self.update_status_bar("Camera thread did not stop gracefully.")
                self.camera_thread.terminate()

        # Cleanup DMD Thread
        if self.dmd_thread.isRunning():
            self.request_cleanup_dmd.emit()
            self.dmd_thread.quit()
            if not self.dmd_thread.wait(3000):
                self.update_status_bar("DMD thread did not stop gracefully.")
                self.dmd_thread.terminate()

        super().closeEvent(event)

    def _on_zoom_slider_changed(self, value):
        # Map slider value (0-100) to scale factor (MIN_ZOOM_SCALE - MAX_ZOOM_SCALE)
        # Linear mapping: scale = min_scale + (value/100) * (max_scale - min_scale)
        scale = PatternCanvas.MIN_ZOOM_SCALE + \
                (value / 100.0) * (PatternCanvas.MAX_ZOOM_SCALE - PatternCanvas.MIN_ZOOM_SCALE)
        self.pattern_canvas.set_absolute_zoom(scale)
        self.zoom_label.setText(f"Zoom: {scale:.2f}x")


    @pyqtSlot(float)
    def _on_canvas_scale_changed(self, scale):
        # Map scale factor back to slider value (0-100)
        # value = ((scale - min_scale) / (max_scale - min_scale)) * 100
        if (PatternCanvas.MAX_ZOOM_SCALE - PatternCanvas.MIN_ZOOM_SCALE) == 0: # Avoid division by zero
            slider_val = 50 
        else:
            slider_val = int( ((scale - PatternCanvas.MIN_ZOOM_SCALE) / \
                              (PatternCanvas.MAX_ZOOM_SCALE - PatternCanvas.MIN_ZOOM_SCALE) ) * 100 )
        
        # Block signals temporarily to avoid feedback loop if slider.setValue triggers valueChanged
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(slider_val)
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"Zoom: {scale:.2f}x")

    def _on_shape_mode_changed(self, shape_text):
        if shape_text == "Square":
            self.shape_size_label.setText("Side Length (px):")
        elif shape_text == "Circle":
            self.shape_size_label.setText("Diameter (px):")
        
        current_size = self.shape_size_spinbox.value()
        if hasattr(self, 'pattern_canvas') and self.pattern_canvas: # Ensure canvas exists
            self.pattern_canvas.set_shape_params(shape_text, current_size)

    def _on_shape_size_changed(self, size):
        current_mode = self.shape_type_combo.currentText()
        if hasattr(self, 'pattern_canvas') and self.pattern_canvas: # Ensure canvas exists
            self.pattern_canvas.set_shape_params(current_mode, size)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyle("Fusion") 
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
