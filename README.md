# Auto-Zooming Cameraman for Basketball Games

This repository contains the code and tools for developing an AI-powered, auto-zooming cameraman system designed for basketball games. The system dynamically adjusts the camera's zoom and focus based on player movements and in-game actions.

## Features
- **Object Detection**: Utilizes YOLOv11 to detect and track players and the basketball in real-time.
- **SAM2 Integration**: Segments players and the basketball to generate masks for more accurate focus.
- **Auto-Zooming**: Automatically adjusts the zoom level to keep relevant actions in view.
- **Post-Processing**: Applies smoothing filters to produce seamless video transitions and viewing experiences.

---

## Project Structure

```plaintext
CIS-5810---Auto-Zooming-Cameraman/
│
├── filters/
│   └── kuwaahara.py           # Kuwahara filter implementation
│
├── nba-commentary-ai/
│   ├── commentary.mp3         # Commentary audio file
│   └── commentary.txt         # Commentary text output
│
├── Object_Detection/
│   ├── Models/                # Pre-trained models (downloaded from provided Drive link)
│   ├── detect.py              # Object detection functions
│   └── utils.py               # Utility functions for detection
│
├── Post-Processing/
│   └── SAM Video Stitching.ipynb  # SAM-based video post-processing
│
├── results/
│   └── example_output.mp4     # Example output video
│
├── team detection/
│   └── team_detection.py      # Team detection code using manual color selection
│
├── UL_streamlit/
│   ├── helper.py              # Helper functions for Streamlit app
│   └── streamlit_app.py       # Streamlit UI for visualization and interaction
│
├── UL_videos/
│   ├── input_frames/          # Input frames for processing
│   └── model_frames/          # Model-generated frames
│
├── zooming/
│   ├── smoothen.py            # Functions for smoothing camera motion
│   └── smoothendarken.py      # Functions for smoothing and darkening frames
│
├── Final_Report_Group_30.pdf  # Final project report
├── requirements.txt           # Python dependencies
├── NBA Game 0021800013.mp4    # Example input video
├── .gitignore
└── README.md                  # This README
```

## How to Run the Project Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VedantZope/CIS-5810---Auto-Zooming-Cameraman.git
   ```

2.	**Change directory and setup the environment:**
      ```bash
      cd CIS-5810---Auto-Zooming-Cameraman
      python3 -m venv myenv
      # On Windows
      myenv\\Scripts\\activate
      # On Unix or MacOS
      source myenv/bin/activate
      ```

3.	**Install the required packages:**
      ```bash
      pip install -r requirements.txt
      ```

4. **Preparation for running the code**
   1.	Download the models from this [link](https://drive.google.com/drive/folders/1e8UovqbuMkoAPPLNhB0fauBhtbLcw3yv?usp=sharing) and put them in the ```Object_detection/Models``` folder
   2. Go to the root of the project and run
      ```bash
      python3 UI_streamlit/streamlit_app.py
      ```
   3.  Then follow the on-screen instructions in the Streamlit UI to interact with the auto-zooming features and view the results.
