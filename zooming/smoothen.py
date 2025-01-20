import numpy as np
import cv2
from collections import deque
from PIL import Image


class SmoothVideoStabilizer:
    def __init__(self, buffer_size=30, max_threshold=0.25, min_threshold=0.5):
        self.buffer_size = buffer_size
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

        # Buffers for dimensions and positions
        self.width_buffer = deque(maxlen=buffer_size)
        self.height_buffer = deque(maxlen=buffer_size)
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)

        # Smoothing parameters
        self.smoothing_factor = 0.1
        self.current_x = None
        self.current_y = None
        self.current_w = None
        self.current_h = None

    def _smooth_value(self, current, target):
        if current is None:
            return target
        return current + (target - current) * self.smoothing_factor

    def _calculate_smooth_dimensions(self, x, y, w, h):
        # Add current values to buffers
        self.x_buffer.append(x)
        self.y_buffer.append(y)
        self.width_buffer.append(w)
        self.height_buffer.append(h)

        # Calculate averages
        avg_x = sum(self.x_buffer) / len(self.x_buffer)
        avg_y = sum(self.y_buffer) / len(self.y_buffer)
        avg_width = sum(self.width_buffer) / len(self.width_buffer)
        avg_height = sum(self.height_buffer) / len(self.height_buffer)

        # Check for sudden changes
        width_diff = abs(w - avg_width) / avg_width
        height_diff = abs(h - avg_height) / avg_height

        # Use average values if change is too drastic
        if width_diff > self.max_threshold:
            w = avg_width
        if height_diff > self.max_threshold:
            h = avg_height

        # Smooth the transitions
        self.current_x = self._smooth_value(self.current_x, x)
        self.current_y = self._smooth_value(self.current_y, y)
        self.current_w = self._smooth_value(self.current_w, w)
        self.current_h = self._smooth_value(self.current_h, h)

        return (int(self.current_x), int(self.current_y),
                int(self.current_w), int(self.current_h))

    def process_frame(self, pil_image, contour):
        # Convert PIL Image to numpy array for contour operations
        frame = np.array(pil_image)

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Get smoothed dimensions
        smooth_x, smooth_y, smooth_w, smooth_h = self._calculate_smooth_dimensions(
            x, y, w, h)

        # Add padding to avoid edge artifacts
        padding = 20
        frame_h, frame_w = frame.shape[:2]

        # Ensure dimensions stay within frame bounds
        smooth_x = max(padding, min(smooth_x, frame_w - smooth_w - padding))
        smooth_y = max(padding, min(smooth_y, frame_h - smooth_h - padding))

        # Crop the numpy array
        cropped = frame[smooth_y:smooth_y+smooth_h,
                        smooth_x:smooth_x+smooth_w]

        # Convert back to PIL Image
        return Image.fromarray(cropped)


# # Usage:
# stabilizer = SmoothVideoStabilizer()
# stabilized_frames = []

# for index, frame in enumerate(test):
#     stabilized_frame = stabilizer.process_frame(frame, contours[index])
#     stabilized_frames.append(stabilized_frame)
