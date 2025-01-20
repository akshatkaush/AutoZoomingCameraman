from pykuwahara import kuwahara
import cv2
import numpy as np
import tempfile 
from stqdm import stqdm

def kuwahara_frame(frame, radius):
    return kuwahara(frame, method='mean', radius=radius)

def kuwahara_process_video(input_path, output_path, kuwahara_param):

    # Create temporary file for output video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Could not open input video")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
        
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame with progress bar
    for _ in stqdm(range(total_frames), desc="Processing video"):
    # while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add your video processing here
        # For example, convert to grayscale and back to BGR
        # processed_frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        processed_frame = kuwahara_frame(frame=frame, radius=kuwahara_param)
        
        # Write processed frame
        out.write(processed_frame)
    
    # Release resources
    cap.release()
    out.release()