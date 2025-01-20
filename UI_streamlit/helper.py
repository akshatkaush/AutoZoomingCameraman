import subprocess
import cv2
from stqdm import stqdm
import os
import shutil
import numpy as np
from PIL import Image

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def make_video_480(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Set new dimensions (keeping aspect ratio)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height
    new_height = 480
    new_width = 640
    
    # Get original FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        resized_frame = cv2.resize(frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_CUBIC)
        out.write(resized_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def make_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def convert_video_h264(input_file, output_file):
    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    subprocess.call(args=f"ffmpeg -y -i {input_file} -c:v libx264 {output_file}".split(" "))

def stitch_frames_to_video(frames,frames_dir, output_video_path,from_dir=False, fps=60):
    if from_dir:
        frames = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        frames.sort(key=lambda p: int(os.path.splitext(p)[0]))

    if len(frames) == 0:
        print("No frames found.")
        return

    # Read the first frame to get the width and height
    if from_dir:
        first_frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    else:
        first_frame = np.asarray(frames[0])
    height, width, layers = first_frame.shape

    # Find the maximum width and height across all frames
    max_width, max_height = 0, 0
    for frame_file in frames:
        if from_dir:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
        else:
            frame = np.asarray(frame_file)

        if frame is None:
            print(f"Error reading frame: {frame_file}")
            continue

        h, w = frame.shape[:2]
        max_width = max(max_width, w)
        max_height = max(max_height, h)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', 'X264', etc.
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (max_width, max_height))

    # Iterate over all frames and write them to the video
    for frame_file in stqdm(frames):
        if from_dir:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
        else:
            frame = np.asarray(frame_file)

        if frame is None:
            print(f"Error reading frame.")
            continue
        # Resize back to the original frame size to create a zoom effect
        zoomed_frame = cv2.resize(frame, (max_width, max_height))
        video.write(zoomed_frame)

    # Release the video writer
    video.release()
    print(f"Video saved at: {output_video_path}")


# Get Video data
def extract_frames(video_path, output_dir):
    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Capture video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    # while True:
    for _ in stqdm(range(total_frames), desc="Extracting frames"):
        # Read a frame
        ret, frame = video.read()

        # Break the loop if no more frames
        if not ret:
            break

         # Save the frame as an image file
        resized_frame = cv2.resize(frame, (1280,790))
        frame_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, resized_frame)
        frame_count+=1

    # Release the video capture object
    video.release()
    print(f"Extracted {frame_count} frames and saved in {output_dir}")
    

def blur_maps(frames,input_video_dir="",object_masks={}, iters=1, blur_kernel_size = (11,11), blur_sigma = 15, save_frames=False, output_dir = "", get_contours=False):
    # Add masks together to create their weighted distribution
    blur_kernel_size = blur_kernel_size  # Adjust for larger or smaller influence areas
    blur_sigma = blur_sigma  # Sigma controls the spread of the influence
    iters = iters

    heatmap_frames = []

    if get_contours:
        contour_dict = {}

    for frame_index,frame in stqdm(enumerate(frames), total=len(frames)):
        video_frame = cv2.imread(os.path.join(input_video_dir, frame))
        h, w = video_frame.shape[:2]

        # Create an empty image with the same size and type as the masks (transparent or black)
        masks_added_total = np.zeros((h, w), dtype=np.float64)

        for object_id, out_mask in object_masks.items():
            out_mask = object_masks[object_id][frame_index][object_id]
            out_frame_idx = frame_index

            # Add gaussian blur
            # influence_map = cv2.GaussianBlur(out_mask[0].astype('uint8'), blur_kernel_size, blur_sigma)
            masks_added_total += out_mask[0].astype('uint8')


            # image = create_mask(out_mask, plt.gca(), obj_id=object_id)
            # im = Image.fromarray((image * 255).astype(np.uint8))
            # plt.imshow(im, alpha=0.7)
            # masks_added_total += image.astype(np.float64)
        # Normalize
        norm_map = cv2.normalize(masks_added_total, None, 0, 255, cv2.NORM_MINMAX)
        norm_map = norm_map.astype(np.uint8)

        for _ in range(iters):
            norm_map = cv2.GaussianBlur(norm_map, blur_kernel_size, blur_sigma)
        norm_map_blurred = norm_map

        # Apply a heatmap color map
        heatmap = cv2.applyColorMap(norm_map_blurred, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_img = Image.fromarray(heatmap)


        # ADD MAX AND BOUNDED BOXING TO CHECK RESULTS.

        bw_video_frame = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(bw_video_frame)

        # Draw centre
        cv2.drawMarker(heatmap, max_loc, color=(0,0,0), markerType=cv2.MARKER_STAR, markerSize=15, thickness=5)

        mean_val, stddev_val = cv2.meanStdDev(bw_video_frame)
        threshold = mean_val + 2.5 * stddev_val

        _, threshold_mask = cv2.threshold(bw_video_frame, threshold.astype(np.uint8).flatten()[0], 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours of the thresholded regions to create the bounding box
        contours, _ = cv2.findContours(threshold_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a bounding rectangle around the largest contour (region of interest)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
            if get_contours:
                contour_dict[frame_index] = largest_contour
            # Draw the rectangle on the heatmap
            cv2.rectangle(heatmap, (x, y), (x + w_roi, y + h_roi), (0,0,0), 2)

        heatmap_img = Image.fromarray(heatmap)

        heatmap_frames.append(heatmap_img)
        # Save the image
        if save_frames:
            heatmap_img.save(f"{output_dir}/{frame_index}.png")

    if get_contours:
        return heatmap_frames, contour_dict
    return heatmap_frames