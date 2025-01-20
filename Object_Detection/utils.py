import cv2
import os
from ultralytics import YOLO
from Object_Detection.detect import detect_persons_only, detect_basketballs


def get_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()

    return tracker


def draw_detections(frame, detections, colors):
    """
    Draw bounding boxes and labels for detections on the frame.
    """
    for det in detections:
        bbox = det['bbox']
        label = det['label']
        conf = det['confidence']
        color = colors.get(label, (255, 255, 255))  # Default to white
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label}: {conf:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_frame(frame, models, state, frame_count, DEVICE, COLORS, PERSON_CONFIDENCE_THRESHOLD, BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD):
    """
    Process a single frame, perform detections and tracking, and draw bounding boxes.
    """
    tracker = state['tracker']
    tracking = state['tracking']
    bbox = state['bbox']
    last_class_id = state['last_class_id']
    tracker_type = state['tracker_type']

    basketball_model = models['basketball_model']
    person_model = models['person_model']

    # Detect objects in the frame
    boxes, confidences, class_ids, class_names = detect_basketballs(frame, basketball_model, BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    person_detections = detect_persons_only(frame, person_model, PERSON_CONFIDENCE_THRESHOLD, DEVICE)

    # Filter detections for 'Basketball' and 'Made-Basket'
    target_classes = ['Basketball', 'Made-Basket']
    filtered_indices = [i for i, cls_id in enumerate(class_ids) if class_names[cls_id] in target_classes]

    boxes = [boxes[i] for i in filtered_indices]
    confidences = [confidences[i] for i in filtered_indices]
    class_ids = [class_ids[i] for i in filtered_indices]

    if boxes:
        # Update last_class_id with the class ID of the first detection
        last_class_id = class_ids[0]

        # If detections are found, initialize/update the tracker
        bbox = boxes[0]
        if tracker is None:
            # Initialize tracker with first detected bounding box
            tracker = get_tracker(tracker_type)
            tracker.init(frame, tuple(bbox))
            tracking = True
            print(f"Tracker initialized at frame {frame_count} with bbox: {bbox}")
        else:
            # Update tracker with new bounding box
            tracker = get_tracker(tracker_type)
            tracker.init(frame, tuple(bbox))
            tracking = True
            print(f"Tracker updated at frame {frame_count} with bbox: {bbox}")
    elif tracking and tracker is not None:
        # If no detections, use tracker to estimate the position
        success, bbox = tracker.update(frame)
        if success:
            # Tracker successfully found the object
            x, y, w, h = [int(v) for v in bbox]
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes = [[x, y, w, h]]
            confidences = [1.0]  # Assign a default confidence
            if last_class_id is not None:
                class_ids = [last_class_id]  # Use last known class ID
            else:
                class_ids = []
        else:
            # Tracker failed to locate the object
            print(f"Tracker lost the object at frame {frame_count}.")
            tracking = False
            tracker = None
            boxes = []
            confidences = []
            class_ids = []
            last_class_id = None  # Reset last_class_id
    else:
        # No detections and not tracking
        boxes = []
        confidences = []
        class_ids = []

    if boxes and class_ids:
        # Prepare basketball detections
        basketball_detections = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'label': class_names[cls_id]
            }
            basketball_detections.append(detection)
    else:
        # Optionally, display that the object is not found
        cv2.putText(frame, 'Object not found', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        basketball_detections = []

    # Combine all detections
    all_detections = basketball_detections + person_detections

    # Draw all detections
    draw_detections(frame, all_detections, colors=COLORS)

    # Update state
    state['tracker'] = tracker
    state['tracking'] = tracking
    state['bbox'] = bbox
    state['last_class_id'] = last_class_id

    # Prepare detections for this frame
    detections = all_detections

    return frame, state, detections

def load_yolo_model(model_path):

    """
    Load YOLOv8 model from a .pt file.
    """
    model = YOLO(model_path)
    return model
