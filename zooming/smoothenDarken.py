import numpy as np
import cv2
from collections import deque
from PIL import Image
from PIL import Image, ImageDraw


class EnhancedVideoStabilizer:
    def __init__(self, buffer_size=30, max_threshold=0.25, min_threshold=0.5, crop_padding=20, zoom_factor=1.2):
        self.buffer_size = buffer_size
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.crop_padding = crop_padding
        # Controls how much to zoom out from the region of interest
        self.zoom_factor = zoom_factor

        # Buffers for dimensions and positions
        self.width_buffer = deque(maxlen=buffer_size)
        self.height_buffer = deque(maxlen=buffer_size)
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)

        # Smoothing parameters - adjust these for smoother transitions
        self.smoothing_factor = 0.05  # Reduced for smoother movement
        self.current_x = None
        self.current_y = None
        self.current_w = None
        self.current_h = None

    def _smooth_value(self, current, target):
        if current is None:
            return target
        # Enhanced smoothing with exponential moving average
        return current + (target - current) * self.smoothing_factor

    def _calculate_smooth_dimensions(self, x, y, w, h):

        # Add padding to the original dimensions
        padded_w = int(w * self.zoom_factor)
        padded_h = int(h * self.zoom_factor)

        # Adjust x and y to center the padded region
        padded_x = x - (padded_w - w) // 2
        padded_y = y - (padded_h - h) // 2

        # Add current values to buffers
        self.x_buffer.append(padded_x)
        self.y_buffer.append(padded_y)
        self.width_buffer.append(padded_w)
        self.height_buffer.append(padded_h)

        # Calculate weighted averages (recent frames have more influence)
        weights = np.linspace(0.5, 1.0, len(self.x_buffer))
        weights = weights / np.sum(weights)

        avg_x = np.average(self.x_buffer, weights=weights)
        avg_y = np.average(self.y_buffer, weights=weights)
        avg_width = np.average(self.width_buffer, weights=weights)
        avg_height = np.average(self.height_buffer, weights=weights)

        # Gradual transition for large position changes
        if abs(x - avg_x) > self.max_threshold * 1024:  # Assuming 1024x1024 frame
            x = avg_x + np.sign(x - avg_x) * self.max_threshold * 1024

        # Smooth the transitions
        self.current_x = self._smooth_value(self.current_x, padded_x)
        self.current_y = self._smooth_value(self.current_y, padded_y)
        self.current_w = self._smooth_value(self.current_w, padded_w)
        self.current_h = self._smooth_value(self.current_h, padded_h)

        # Add additional padding for crop
        final_x = int(self.current_x - self.crop_padding)
        final_y = int(self.current_y - self.crop_padding)
        final_w = int(self.current_w + 2 * self.crop_padding)
        final_h = int(self.current_h + 2 * self.crop_padding)

        return (final_x, final_y, final_w, final_h)

    def darken_borders(self, pil_image, border_width=20):
        # Create a gradient border effect
        new_image = Image.new('RGBA', pil_image.size)
        draw = ImageDraw.Draw(new_image)
        width, height = pil_image.size

        # Create a gradient mask
        for i in range(border_width):
            opacity = int(128 * (i / border_width))  # Gradual transparency
            draw.rectangle(
                [i, i, width-i, height-i],
                outline=(0, 0, 0, opacity)
            )

        # Composite the original image with the border
        result = Image.alpha_composite(pil_image.convert('RGBA'), new_image)
        return result

    def process_frame(self, pil_image, contour):
        # Get the smoothed frame
        frame = np.array(pil_image)
        x, y, w, h = cv2.boundingRect(contour)

        smooth_x, smooth_y, smooth_w, smooth_h = self._calculate_smooth_dimensions(
            x, y, w, h)

        # Add padding for smooth transitions
        padding = 40  # Increased padding
        frame_h, frame_w = frame.shape[:2]

        smooth_x = max(padding, min(smooth_x, frame_w - smooth_w - padding))
        smooth_y = max(padding, min(smooth_y, frame_h - smooth_h - padding))

        cropped = frame[smooth_y:smooth_y+smooth_h,
                        smooth_x:smooth_x+smooth_w]

        # Convert to PIL and apply border darkening
        pil_cropped = Image.fromarray(cropped)

        darkened = self.darken_borders(pil_cropped)

        # Return back to RGB
        final = darkened.convert('RGB')
        # final = pil_cropped.convert('RGB')

        return final


# # Initialize the stabilizer
# stabilizer = EnhancedVideoStabilizer(
#     buffer_size=30,  # Increase for smoother but slower transitions
#     max_threshold=0.25,  # Adjust for maximum allowed sudden changes
#     min_threshold=0.5  # Minimum threshold for changes
# )

# # Process your video frames
# stabilized_frames_new = []
# for index, frame in enumerate(test):
#     stabilized_frame = stabilizer.process_frame(frame, contours[index])
#     stabilized_frames_new.append(stabilized_frame)
