import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('Gate_DNT.pt')  

# Function to calculate the center of a bounding box
def get_center(x, y, w, h):
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    return (center_x, center_y)

# Function to process a frame and apply YOLOv8
def process_frame(frame):
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Center of the screen
    screen_center = (width // 2, height // 2)

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Extract the bounding boxes, confidences, and classes from the results and move to CPU
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Minimum confidence threshold
    min_confidence = 0.8

    # Check if any detections were made
    if len(bboxes) > 0:
        # Loop over the detections
        for i, bbox in enumerate(bboxes):
            if confidences[i] >= min_confidence:  # Apply the confidence threshold
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate the width and height of the bounding box
                w = x2 - x1
                h = y2 - y1

                # Get the center of the bounding box
                box_center = get_center(x1, y1, w, h)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw a circle at the center of the bounding box
                cv2.circle(frame, box_center, 5, (0, 0, 255), -1)

                # Draw a circle at the center of the screen
                cv2.circle(frame, screen_center, 5, (255, 0, 0), -1)

                # Draw a line between the two centers
                cv2.line(frame, screen_center, box_center, (255, 255, 0), 2)

                # Determine movement direction
                direction_vertical = "Move Up" if box_center[1] < screen_center[1] else "Move Down"
                direction_horizontal = "Move Left" if box_center[0] < screen_center[0] else "Move Right"

                # Display the directions on the image
                cv2.putText(frame, direction_vertical, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, direction_horizontal, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # No detections made
        cv2.putText(frame, "No detections made", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Initialize video capture (0 for default camera, replace with video path if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process the frame
    processed_frame = process_frame(frame)
    
    # Display the frame
    cv2.imshow('Frame', processed_frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()