import cv2
import numpy as np
import os 
import pandas as pd
from stqdm import stqdm


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_centroid_centre(contours):
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

    average_x = sum(x for x, y in centroids) / len(centroids)
    average_y = sum(y for x, y in centroids) / len(centroids)
    center = (int(average_x), int(average_y))

    return center


def count_pixels_in_range(image, lower_bound, upper_bound):
    """Count the number of pixels within a specified color range."""
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return np.count_nonzero(mask)

def extract_players(frame, bboxes):

    team = []
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        team.append( frame[x1:x2,y1:y2])
    
    return team



def classify_teams_by_color(frame, bboxes, team_colors, blue_lower, blue_upper, white_lower, white_upper):
    """Classify bounding boxes into teams based on pixel counts in color ranges."""
    team_colors = {
        0: (255, 0, 0),  # Blue
        1: (255, 255, 255),  # White
    }

    blue_heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)  # Use a single channel
    white_heatmap = np.zeros_like(frame[:, :, 0], dtype=np.float32)

    blue_centroids = []
    white_centroids = []

    people_bbox = [i['bbox'] for i in bboxes['detections'] if i['label'] == 'person']

    # Define color ranges in HSV
    blue_lower = np.array([100, 150, 50])  # Adjust these ranges for specific blue
    blue_upper = np.array([140, 255, 255])
    white_lower = np.array([0, 0, 200])  # Adjust for specific white brightness
    white_upper = np.array([180, 30, 255])

    # Prepare to draw on the frame
    labeled_frame = frame.copy()

    detected_people = { 0 : [], 1 : []}

    for bbox in people_bbox:
        x1, y1, x2, y2 = bbox
        cropped = frame[y1:y2, x1:x2]

        # Convert the cropped region to HSV
        hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Count pixels in blue and white ranges
        blue_count = count_pixels_in_range(hsv_cropped, blue_lower, blue_upper)
        white_count = count_pixels_in_range(hsv_cropped, white_lower, white_upper)

        # Classify based on the dominant color
        if blue_count > white_count:
            label = 0  # Team 0 (Blue)
            heatmap = blue_heatmap
        elif white_count >= blue_count:
            label = 1  # Team 1 (White)
            heatmap = white_heatmap
        else:
            # Handle ambiguous cases (e.g., no dominant color)
            label = -1
        
        # Add heat to the center of the bounding box (can also add at the whole bbox if preferred)
        # Add heat over the entire bounding box area
        heatmap[y1:y2, x1:x2] += 1   # Increment the heat at the center

        # Draw the bounding box and label
        if label != -1:
            color = team_colors[label]
            cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(labeled_frame, f"Team {label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            
            detected_people[label].append(cropped)

    basketball_bb = [i['bbox'] for i in bboxes['detections'] if i['label'] == 'Basketball']
    if len(basketball_bb) > 0:
        basketball_bb = basketball_bb[0]

    baskbetball_centre = (None,None)
    if basketball_bb!=[]:
        x1,y1,x2,y2 = basketball_bb
        color = (0,0,255)
        cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(labeled_frame, f"Basketball", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        
        baskbetball_centre = x1+(abs(x1-x2)//2),y1+(abs(y1-y2)//2)

    # Apply Gaussian blur to create a smoother heatmap
    blue_heatmap = cv2.GaussianBlur(blue_heatmap, (101, 101), 51)
    white_heatmap = cv2.GaussianBlur(white_heatmap, (101, 101), 51)

    # Normalize the heatmaps to 0-255 range for display
    blue_heatmap = cv2.normalize(blue_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    white_heatmap = cv2.normalize(white_heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Get contours and centroids
    blue_contours, _ = cv2.findContours(blue_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_contours, _ = cv2.findContours(white_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_centre = get_centroid_centre(blue_contours)
    white_centre = get_centroid_centre(white_contours)

    blue_heatmap = cv2.circle(blue_heatmap.astype(np.uint8), blue_centre, radius=3, color=team_colors[0], thickness=10)
    white_heatmap = cv2.circle(white_heatmap.astype(np.uint8), white_centre, radius=3, color=team_colors[1], thickness=10)

    blue_center_image = np.zeros_like(frame[:, :, 0], dtype=np.float32)
    white_center_image = np.zeros_like(frame[:, :, 0], dtype=np.float32)
    blue_center_image = cv2.circle(blue_center_image, blue_centre, radius=3, color=team_colors[0], thickness=10)
    white_center_image = cv2.circle(white_center_image, white_centre, radius=3, color=team_colors[1], thickness=10)
               

    # Stack the heatmaps for display
    heatmap_image = cv2.merge([blue_heatmap, white_heatmap, np.zeros_like(blue_heatmap)])

    # Overlay heatmap on original frame for visualization
    # heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_image.astype(np.uint8), 0.3, 0)

    return labeled_frame, heatmap_image, blue_centre, white_centre, baskbetball_centre, detected_people


def run_team_detection(frames_path, bounding_boxes, team_colors, blue_lower, blue_upper, white_lower, white_upper ):

    blue_centre_list_x = []
    blue_centre_list_y = []
    white_centre_list_x = []
    white_centre_list_y = []
    basketball_centre_list_x = []
    basketball_centre_list_y = []

    frames = [i for i in os.listdir(frames_path) if i.endswith('.jpg')]
    frames.sort()


    for index, frame in stqdm(enumerate(frames), total=len(frames)):
        
        img = cv2.imread(f'{frames_path}/{frame}')
        labeled_frame, heatmap_image, blue_centre, white_centre, baskbetball_centre, detected_people = classify_teams_by_color(frame = img, bboxes=bounding_boxes[index], team_colors = team_colors,blue_lower = blue_lower, blue_upper = blue_upper,white_lower = white_lower,white_upper = white_upper)
        blue_centre_list_x.append(blue_centre[0])
        blue_centre_list_y.append(blue_centre[1])
        white_centre_list_x.append(white_centre[0])
        white_centre_list_y.append(white_centre[1])
        basketball_centre_list_x.append(baskbetball_centre[0])
        basketball_centre_list_y.append(baskbetball_centre[1])

        # cv2.imwrite(f"/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/team_labelled_frames/{index}.jpg",labeled_frame)
        # cv2.imwrite(f"/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/team_heatmap_labelled_frames/{index}.jpg",heatmap_image)

        if index == 10:
            for img_index,i in enumerate(detected_people[0]):
                cv2.imwrite(f"/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/teams/0/{img_index}.jpg",i)
            for img_index,i in enumerate(detected_people[1]):
                cv2.imwrite(f"/Users/umangsharma/Documents/GitHub/CIS-5810-Auto-Zooming-Cameraman/teams/1/{img_index}.jpg",i)
        if index == 0:
            stacked_heatmap = heatmap_image
        else:
            stacked_heatmap = cv2.addWeighted(stacked_heatmap, 0.5, heatmap_image, 0.5, 0)

    # Normalize stacked heatmap to 0-255 for better overlay (important!)
    stacked_heatmap = cv2.normalize(stacked_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    stacked_heatmap = stacked_heatmap.astype(np.uint8)
    
    court_image = cv2.imread('basketball_field.jpg') # Replace with the path
    court_image = cv2.resize(court_image, (stacked_heatmap.shape[1], stacked_heatmap.shape[0]))
    overlay_image = cv2.addWeighted(court_image, 0.3, stacked_heatmap, 0.7, 0) # Adjust weights as needed
    cv2.imwrite("team_movement.jpg", overlay_image)

    # Create dataframe and store 
    positions_df = pd.DataFrame(
    {
        'blue_x': blue_centre_list_x,
        'blue_y': blue_centre_list_y,
        'white_x': white_centre_list_x,
        'white_y': white_centre_list_y,
        'basketball_x': basketball_centre_list_x,
        'basketball_y':basketball_centre_list_y
    })
    positions_df['distance_to_blue'] = calculate_distance(
        positions_df['basketball_x'], positions_df['basketball_y'],
        positions_df['blue_x'], positions_df['blue_y']
    )

    positions_df['distance_to_white'] = calculate_distance(
        positions_df['basketball_x'], positions_df['basketball_y'],
        positions_df['white_x'], positions_df['white_y']
    )

    # Create column showing which team is closer to ball
    positions_df['closer_to_ball'] = np.where(
        positions_df['distance_to_blue'] < positions_df['distance_to_white'],
        'blue',
        'white'
    )
    positions_df['closer_to_ball'] = np.where(
        positions_df['basketball_y']!="",
        positions_df['closer_to_ball'],
        ""
    )

    positions_df.to_excel("positions_df.xlsx")



    