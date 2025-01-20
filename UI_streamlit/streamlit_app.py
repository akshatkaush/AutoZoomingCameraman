
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tempfile
import numpy as np
import cv2
import base64
import streamlit as st
import os 
import subprocess
import json
from PIL import Image
from PIL import Image, ImageDraw

# Get the parent directory of 'main'
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from filters.kuwahara import kuwahara_process_video
import helper
import shutil

# Import object detection model - YOLOv8 
from Object_Detection.YOLO_v8_model_helper import *

# Import SmoothenZoom 
from zooming.smoothen import SmoothVideoStabilizer
from zooming.smoothenDarken import EnhancedVideoStabilizer
from zooming.zoom_final import Stabilizer

# Import team detection capability 
from team_detection.team_detection import *
from PIL import ImageColor
import pandas as pd


def init_session_state():
    """Initialize session state variables"""
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'segmentation_done' not in st.session_state:
        st.session_state.segmentation_done = False
    if 'heatmap_done' not in st.session_state:
        st.session_state.heatmap_done = False
    if 'filter_done' not in st.session_state:
        st.session_state.filter_done = False
    if 'processing_started' not in st.session_state:
        st.session_state.processing_started = False

def upload_video():
    """Handle video upload"""
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        st.session_state.video_file = uploaded_file
        
        if uploaded_file.name == "NBA Game 0021800013.mp4":
            # print("DEMO FILE")
            st.session_state.demo = True

        # save uploaded video to disc
        if "UI_videos" not in os.listdir():
            os.mkdir("./UI_videos")

        temp_file_to_save = './UI_videos/input.mp4'
        output_path = './UI_videos/input480p.mp4'
        helper.write_bytesio_to_file(temp_file_to_save, uploaded_file)
        # helper.make_video_480(temp_file_to_save, output_path)
        st.session_state.input_video_file = temp_file_to_save
        return True
    return False

def run_segmentation(model_type, video_file):
    """Simulate segmentation processing"""

    tfile_model = tempfile.NamedTemporaryFile(delete=False)
    tfile_model.write(video_file.read())

    outputfile = f"./UI_videos/model_{model_type}_output.mp4"

    with st.spinner("Running detection..."):

        if st.session_state.demo:
            print("frames detected")
            frame_detections = np.load("yolo_frame_detections.npy", allow_pickle=True)

        elif model_type =="YOLOv11":
            print("YOLO")

            frame_detections = YOLO(INPUT_VIDEO=st.session_state.input_video_file, OUTPUT_VIDEO=outputfile)

        st.session_state.yolo_frame_detections = frame_detections
        st.success("Detection completed!")
    
    st.session_state.segmentation_done = True
    # np.save("yolo_frame_detections.npy" ,frame_detections)

    #Extract INPUT Frames
    path = "./UI_videos/input_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=st.session_state.input_video_file,output_dir=path)

    # Extract SEGMENTATION frames if needed
    path = "./UI_videos/model_frames/"
    helper.make_path(path=path)
    helper.extract_frames(video_path=outputfile,output_dir=path)

    convertedVideo_model = f"./UI_videos/model_{model_type}_output_h264.mp4"
    helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo_model)

    st.session_state.modelh264_video_file = convertedVideo_model

    if st.session_state.debug:
        st.video(convertedVideo_model)

    st.subheader("Step 1.5: Running team Detection")
    st.toast('Running')
    team_colors = {
        0: [255,0,0],  # Blue
        1: [255,255,255],  # White
    }
    blue_lower = np.array([100, 150, 50])  # Adjust these ranges for specific blue
    blue_upper = np.array([140, 255, 255])
    white_lower = np.array([0, 0, 200])  # Adjust for specific white brightness
    white_upper = np.array([180, 30, 255])

    run_team_detection(frames_path = "./UI_videos/input_frames/", bounding_boxes = frame_detections, team_colors = team_colors,blue_lower = blue_lower, blue_upper = blue_upper,white_lower = white_lower,white_upper = white_upper)
    st.session_state.team_detection_done = True
    st.success("Teams Detected!")

def generate_heatmap(model_type):
    """Generate and display heatmap"""
    
    outputfile = f"./UI_videos/model_{model_type}_merged.mp4"

    with st.spinner("Generating heatmap..."):
        if model_type=='YOLOv11':
            WEIGHTS = {'person': 10, 'Basketball': 50}
            blurred_heatmaps, contours = generate_heatmap_video(frame_detections= st.session_state.yolo_frame_detections,video_path= st.session_state.modelh264_video_file,
                                    output_path=outputfile, return_heatmaps = True, weight_mapping=WEIGHTS)
            

            
            # with open('merged_frames_heatmap_yolov8.txt', 'w') as convert_file: 
            #     convert_file.write(json.dumps(heatmaps))

            # np.save('merged_frames_heatmap_yolov8.npy', blurred_heatmaps)

            # Store the Contours for use later 
            st.session_state.contours = contours
            
            # Extract MERGED frames if needed
            path = "./UI_videos/model_merged_frames/"
            helper.make_path(path=path)
            helper.extract_frames(video_path=outputfile,output_dir=path)

            # Store frame names in memory 
            frame_names = [
                p for p in os.listdir(path)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]

    convertedVideo_blurred = f"./UI_videos/model_{model_type}_blurred_h264.mp4"
    helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo_blurred)
    st.session_state.modelh264_video_file = convertedVideo_blurred

    if st.session_state.debug:
        st.video(convertedVideo_blurred)

    st.session_state.heatmap_done = True

def apply_filter(filter_type, video_file):
    """ First Zoom""" 

    output_zoomed_file = "./UI_videos/filter_output.mp4"

    with st.spinner(f"Automatically Zooming..."):
        # Call Zoom 
        # Initialize the stabilizer
        stabilizer = Stabilizer(
            buffer_size=30,  # Increase for smoother but slower transitions
        )
        stabilizer.position_threshold = 3  # Adjust for maximum allowed sudden changes
        stabilizer.size_threshold = 8   # Minimum threshold for changes
        stabilizer_frames = []
        # Get original input frames
        input_path =  "./UI_videos/input_frames/"
        frame_names = [
            p for p in os.listdir(input_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        st.toast("ZOOMING")
        for index,frame_name in stqdm(enumerate(frame_names), total=len(frame_names)):
            frame = cv2.imread(f"{input_path}/{frame_name}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Cnvrt frame to pillow image 
            frame_pil = Image.fromarray(frame_rgb)
            if index not in st.session_state.contours:
                print("ERROR - FRAME NOT FOUND ", index)
                continue
            stabilized_frame = stabilizer.process_frame(frame_pil, st.session_state.contours[index])
            stabilizer_frames.append(cv2.cvtColor(np.array(stabilized_frame),cv2.COLOR_BGR2RGB))

            if index==0:
                cv2.imwrite("test.jpeg", cv2.cvtColor(np.array(stabilized_frame),cv2.COLOR_BGR2RGB))
                

    # Create Zoomed video
    helper.stitch_frames_to_video(frames=stabilizer_frames,frames_dir="",output_video_path=output_zoomed_file, from_dir=False)

    converted_zoomed_Video = "./UI_videos/filter_zoomed_h264.mp4"
    helper.convert_video_h264(input_file=output_zoomed_file, output_file=converted_zoomed_Video)

    if st.session_state.debug:
        st.video(converted_zoomed_Video)


    """Apply selected filter to video"""

    if filter_type!="None":

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        outputfile = "./UI_videos/filter_output.mp4"

        with st.spinner(f"Applying {filter_type} filter..."):
            # Process the video
            if filter_type=="Kuwahara":
                kuwahara_process_video(input_path=converted_zoomed_Video,
                                                    output_path = outputfile, kuwahara_param = st.session_state.kuwahara_param)
            else:
                st.text("No filter selected.")

            st.success(f"{filter_type} filter applied! - Converting to Appropriate Codec.")
            st.session_state.filter_done = True
        
        convertedVideo = "./UI_videos/filter_output_h264.mp4"
        helper.convert_video_h264(input_file=outputfile, output_file=convertedVideo)

        # # Extract frames for analysis later

        path = "./UI_videos/filter_frames/"
        helper.make_path(path=path)
        helper.extract_frames(video_path=convertedVideo,output_dir=path)

        if st.session_state.debug:
            st.video(convertedVideo)
    
def show_video_details(video_file):
    """Display video metadata"""
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video = cv2.VideoCapture(tfile.name)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    if st.session_state.debug:
        st.subheader("Video Details")
        col1, col2, col3 = st.columns(3)
        col1.metric("FPS", f"{fps:.2f}")
        col2.metric("Frames", frame_count)
        col3.metric("Duration (s)", f"{duration:.2f}")
        st.video(video_file)

def hometab():
    if st.session_state.video_file is not None:
        st.header("Processing Pipeline")

        # Show video details
        show_video_details(st.session_state.video_file)

        if st.session_state.processing_started:
            # Run segmentation
            st.subheader("Step 1: Detection")
            if not st.session_state.segmentation_done:
                run_segmentation(st.session_state.segmentation_model, st.session_state.video_file)

            # Generate heatmap
            if st.session_state.segmentation_done:
                st.subheader("Step 2: Heatmap Generation")
                if not st.session_state.heatmap_done:
                    generate_heatmap(st.session_state.segmentation_model)

            # Zoom and Apply filter if selected
            if st.session_state.heatmap_done:
                st.subheader("Step 3: Zooming and Applying Filter")
                if not st.session_state.filter_done:
                    apply_filter(st.session_state.filter_type, st.session_state.video_file)
                    # st.session_state.filtered_video = output_video

            # Final output
            if st.session_state.heatmap_done:
                st.subheader("Final Output")
                # st.download_button(
                #     label="Download Processed Video",
                #     data=st.session_state..getvalue(),
                #     file_name="processed_video.mp4",
                #     mime="video/mp4"
                # )
    else:
        st.info("Please upload a video file to begin processing")

def commentaryTab(video_name):
    st.header("Commentary")
    if st.session_state.processing_started == True:

        with open(f"nba-commentary-ai/{video_name}.txt", "r") as file:
            commentary = file.read()
        text_stream = [i for i in commentary]
        with st.spinner("Generating summary"):
            st.write_stream(text_stream)

        # st.write(commentary)
        #audio
        audio_file = open(f"nba-commentary-ai/{video_name}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.write("Please start processing the video in the first tab.")


def sidebar():
    st.header("Processing Steps")
    # Step 1: Video Upload
    st.subheader("1. Upload Video")
    video_name = None
    if upload_video():
        video_name = st.session_state.video_file.name
        video_name = video_name.rsplit('.', 1)[0]  # Remove file extension
        st.success(f"Video uploaded successfully: {video_name}")

        if st.session_state.demo:
            st.info("Demo video detected")

    # Debug 
    debug = st.checkbox(label="Debug")
    st.session_state.debug = debug

    # Step 2: Segmentation Settings
    st.subheader("2. Detection Settings")
    segmentation_model = st.selectbox(
        "Select Detection Model",
        ["YOLOv11", "SAM2"]
    )
    st.session_state.segmentation_model = segmentation_model

    # Step 3: Heatmap Settings
    st.subheader("3. Team Settings")
    
    col1, col2 = st.columns(2)

    with col1:
        team_1_color = st.color_picker(label = "Team 1", value="#0000ff")
        st.session_state.team1color = ImageColor.getcolor(team_1_color, "RGB")

    with col2:
        team_2_color = st.color_picker(label = "Team 2", value="#ffffff")
        st.session_state.team2color = ImageColor.getcolor(team_2_color, "RGB")



    # Step 4: Filter Settings
    st.subheader("4. Filter Settings (Optional)")
    filter_type = st.selectbox(
        "Select Filter",
        ["None", "Blur", "Sharpen", "Grayscale", "Kuwahara"]
    )
    st.session_state.filter_type = filter_type
    if filter_type == "Kuwahara":
        kuwahara_param = st.number_input(label="Kuwahara radius", min_value=1, max_value=10, step=1, format="%i")
        st.session_state.kuwahara_param = kuwahara_param

    # Start Processing Button
    if st.button("Start Processing") and st.session_state.video_file is not None:
        st.session_state.processing_started = True

    return video_name

def game_analysis():
    st.header("Game Analysis")
    st.info('Detecting teams')

    if st.session_state.segmentation_done:
        st.subheader("Detected Teams ")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team one")
            dir1 = "/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/teams/0"
            images1 = [os.path.join(dir1, img) for img in os.listdir(dir1)]
            st.image(images1, width=50, use_container_width=False)
        
        with col2:
        # Second row container
            st.subheader("Team two")
            dir2 = "/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/teams/1"
            images2 = [os.path.join(dir2, img) for img in os.listdir(dir2)]
            st.image(images2, width=50, use_container_width=False)

        st.header("Average Team movement over time")
        heatmap_img = cv2.imread("team_movement.jpg")
        st.image(heatmap_img)

        st.header("Possession %")

        df = pd.read_excel("positions_df.xlsx")
        white_perc = df['closer_to_ball'].value_counts()['white']/ len(df)
        blue_perc = df['closer_to_ball'].value_counts()['blue']/ len(df)

            
        col1, col2 = st.columns(2)

        with col1:
            st.header("Blue")
            st.metric(label="Team one", value=f"{blue_perc*100:.2f} %", label_visibility="hidden")

        with col2:
            st.header("White")
            st.metric(label="Team two", value=f"{white_perc*100:.2f}%", label_visibility="hidden")

        st.header("Team Trajectory over time")

        frame_mumber = st.number_input(label="Frame number",value=456, min_value=31, max_value=639, step=1, format="%i")
        trail_len = 30
        alpha_values = [0.8 ** i for i in range(trail_len)]
        alpha_values = [i/max(alpha_values) for i in alpha_values]
        size = [2500*i for i in alpha_values]

        # Set the style and create a dark background
        sns.set_theme(style="whitegrid", palette="pastel")

        # Create figure with specific size
        fig = plt.figure(figsize=(12, 8))

        # Create custom color palettes
        blue_palette = sns.dark_palette("#79C", n_colors=trail_len)
        green_palette = sns.dark_palette("seagreen", n_colors=trail_len)

        palette = sns.color_palette("bright")

        sns.scatterplot( 
            x=df['blue_x'].iloc[frame_mumber-trail_len:frame_mumber],
            y=df['blue_y'].iloc[frame_mumber-trail_len:frame_mumber],
            color = palette[0],
            alpha = alpha_values,
            s = size
        )
        sns.scatterplot(
            x=df['white_x'].iloc[frame_mumber-trail_len:frame_mumber],
            y=df['white_y'].iloc[frame_mumber-trail_len:frame_mumber],
            color = palette[1],
            alpha = alpha_values,
            s = size
        )

        # Plot blue team with gradient colors
        # for i in range(trail_len):
        #     sns.scatterplot(
        #         x=[df['blue_x'].iloc[frame_mumber+i]],
        #         y=[df['blue_y'].iloc[frame_mumber+i]],
        #         color=blue_palette[i],
        #         s=size[i],
        #         label="Team 0" if i == 0 else "",
        #     )

        # # Plot white team with gradient colors
        # for i in range(trail_len):
        #     sns.scatterplot(
        #         x=[df['white_x'].iloc[frame_mumber+i]],
        #         y=[df['white_y'].iloc[frame_mumber+i]],
        #         color=green_palette[i],
        #         s=size[i],
        #         label="Team 1" if i == 0 else "",
        #     )

        # Set fixed axis limits
        plt.xlim(0, 1280)
        plt.ylim(0, 780)

        # Customize the plot
        plt.title("Player Movement Trails", pad=20, fontsize=16, fontweight='bold')
        plt.xlabel("Court Width (pixels)", fontsize=12)
        plt.ylabel("Court Height (pixels)", fontsize=12)

        # Customize grid and legend
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=10, framealpha=0.8)

        st.pyplot(fig)




def main():
    st.title("Computer Vision Final Project Group 30 Dashboard")
    init_session_state()

    # Tabs
    home_tab, commentary_tab, game_analysis_tab = st.tabs(["Home", "Commentary","Game Analysis"])

    # Add content to first tab
    with home_tab:
        hometab()

    # Left sidebar for step selection
    with st.sidebar:
        video_name = sidebar()

    # Add content to second tab
    with commentary_tab:
        commentaryTab(video_name)

    with game_analysis_tab:
        game_analysis()


if __name__ == "__main__":
    main()
