import cv2
import numpy as np

def detect_persons(frame, model, confidence_threshold, DEVICE):
    """
    Detect persons in a single frame using YOLOv11.
    Returns a list of detections with bounding boxes and labels.
    """
    results = model(frame, imgsz=1280, device=DEVICE, classes=[0])
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy())
            class_name = model.names[cls_id]
            if conf >= confidence_threshold:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': class_name,
                    'confidence': conf
                })
    return detections

def detect_persons_only(frame, person_model, confidence_threshold, device):
    """
    Detect persons in the frame.
    """
    detections = detect_persons(frame, person_model, confidence_threshold, device)
    return detections

def detect_basketballs(frame, model, BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD,  imgsz=1280):
    """
    Detect basketballs and made baskets in a single frame using YOLOv8.
    Returns bounding boxes, confidences, and class IDs.
    """
    results = model(frame, imgsz=imgsz)
    boxes = []
    confidences = []
    class_ids = []
    class_names = model.names  # Get class names from the model

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy().item())
            cls_id = int(box.cls.cpu().numpy().item())
            if conf >= BALL_CONFIDENCE_THRESHOLD:
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(conf)
                class_ids.append(cls_id)

    # Apply Non-Max Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, BALL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    filtered_boxes = []
    filtered_confidences = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        filtered_boxes.append(boxes[i])
        filtered_confidences.append(confidences[i])

    return filtered_boxes, filtered_confidences, class_ids, class_names