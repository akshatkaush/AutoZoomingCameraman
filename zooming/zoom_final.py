import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from collections import deque

from PIL import Image

class Stabilizer:
    def __init__(self, buffer_size=30):
        self.buffer_size = buffer_size
        self.aspect_ratio = 16/9
        
        # Position and size buffers
        self.x_buffer = deque(maxlen=buffer_size)
        self.y_buffer = deque(maxlen=buffer_size)
        self.size_buffer = deque(maxlen=buffer_size)
        
        # Movement thresholds
        self.position_threshold = 2.0  # pixels
        self.size_threshold = 5.0      # pixels
        
        # 8D state: [x, vx, ax, y, vy, ay, size, size_rate]
        self.kf = KalmanFilter(dim_x=8, dim_z=3)
        
        # Initial state
        self.kf.x = np.zeros(8)
        
        # State transition matrix
        dt = 1.0
        self.kf.F = np.zeros((8, 8))
        # Position states
        self.kf.F[0:3, 0:3] = np.array([[1, dt, 0.5*dt*dt],
                                       [0, 1, dt],
                                       [0, 0, 1]])
        # Velocity states
        self.kf.F[3:6, 3:6] = np.array([[1, dt, 0.5*dt*dt],
                                       [0, 1, dt],
                                       [0, 0, 1]])
        # Size states
        self.kf.F[6:8, 6:8] = np.array([[1, dt],
                                       [0, 1]])
        
        # Measurement function
        self.kf.H = np.zeros((3, 8))
        self.kf.H[0, 0] = 1  # x measurement
        self.kf.H[1, 3] = 1  # y measurement
        self.kf.H[2, 6] = 1  # size measurement
        
        # Measurement noise
        self.kf.R = np.eye(3) * 2000  # Increased for more stability
        
        # Process noise
        self.kf.Q = np.eye(8) * 0.00001  # Decreased for smoother tracking
        
        # Initial covariance
        self.kf.P = np.eye(8) * 1000
        
        # Previous values
        self.prev_x = None
        self.prev_y = None
        self.prev_size = None
        
        # Smoothing factors
        self.pos_smooth_factor = 0.98    # Increased for more stability
        self.size_smooth_factor = 0.99   # Increased for more stability
        
        # Transition parameters
        self.max_size_change_rate = 0.03  # Maximum 3% change per frame
        self.target_size = None
        self.current_size = None
        
    def moving_average(self, buffer, value):
        buffer.append(value)
        weights = np.linspace(1, 2, len(buffer))
        return np.average(buffer, weights=weights)
        
    def smooth_value(self, prev_val, new_val, smooth_factor, threshold):
        if prev_val is None:
            return new_val
            
        # Check if movement is below threshold
        if abs(new_val - prev_val) < threshold:
            return prev_val
            
        return prev_val * smooth_factor + new_val * (1 - smooth_factor)
    
    def smooth_size_transition(self, current_size, target_size):
        if current_size is None:
            return target_size
            
        # Check if size change is below threshold
        if abs(target_size - current_size) < self.size_threshold:
            return current_size
            
        # Calculate maximum allowed change
        max_change = current_size * self.max_size_change_rate
        size_diff = target_size - current_size
        
        # Limit the size change
        if abs(size_diff) > max_change:
            if size_diff > 0:
                return current_size + max_change
            else:
                return current_size - max_change
        
        return target_size
    
    def calculate_crop_dimensions(self, frame_height, frame_width, target_height):
        """Calculate crop dimensions maintaining 16:9 aspect ratio"""
        target_width = int(target_height * self.aspect_ratio)
        
        # Ensure dimensions don't exceed frame size
        if target_width > frame_width:
            target_width = frame_width
            target_height = int(target_width / self.aspect_ratio)
        
        if target_height > frame_height:
            target_height = frame_height
            target_width = int(target_height * self.aspect_ratio)
            
        return target_width, target_height
    
    def process_frame(self, frame, contour):

        # Convert Image to array 
        frame = np.array(frame)

        frame_height, frame_width = frame.shape[:2]
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center
        center_x = x + w/2
        center_y = y + h/2
        
        # Apply moving average to positions and size
        center_x = self.moving_average(self.x_buffer, center_x)
        center_y = self.moving_average(self.y_buffer, center_y)
        current_size = self.moving_average(self.size_buffer, h)
        
        # Kalman prediction and update
        self.kf.predict()
        measurement = np.array([center_x, center_y, current_size])
        self.kf.update(measurement)
        
        # Get predicted values
        predicted_x = self.kf.x[0]
        predicted_y = self.kf.x[3]
        predicted_size = self.kf.x[6]
        
        # Apply additional smoothing with thresholds
        if self.prev_x is not None:
            predicted_x = self.smooth_value(self.prev_x, predicted_x, 
                                         self.pos_smooth_factor, self.position_threshold)
            predicted_y = self.smooth_value(self.prev_y, predicted_y, 
                                         self.pos_smooth_factor, self.position_threshold)
            predicted_size = self.smooth_value(self.prev_size, predicted_size, 
                                            self.size_smooth_factor, self.size_threshold)
        
        # Smooth size transitions
        if self.current_size is None:
            self.current_size = predicted_size
        
        self.current_size = self.smooth_size_transition(self.current_size, predicted_size)
        
        # Update previous values
        self.prev_x = predicted_x
        self.prev_y = predicted_y
        self.prev_size = self.current_size
        
        # Calculate dimensions maintaining aspect ratio
        target_width, target_height = self.calculate_crop_dimensions(
            frame_height, frame_width, int(self.current_size))
        
        # Calculate new frame bounds
        new_x = int(predicted_x - target_width/2)
        new_y = int(predicted_y - target_height/2)
        
        # Ensure bounds are within frame with padding
        padding = 50
        new_x = max(padding, min(new_x, frame_width - target_width - padding))
        new_y = max(padding, min(new_y, frame_height - target_height - padding))
        
        # Final boundary check
        new_x = max(0, min(new_x, frame_width - target_width))
        new_y = max(0, min(new_y, frame_height - target_height))
        
        try:
            cropped = frame[new_y:new_y+target_height, new_x:new_x+target_width]
            return cropped
        except:
            return frame
