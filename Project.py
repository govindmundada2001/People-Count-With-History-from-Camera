import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
from datetime import datetime

# Create a SQLite database to store the history of detected individuals
conn = sqlite3.connect('object_detection.db')
c = conn.cursor()

# Create a table to store the history of detected individuals
c.execute('''CREATE TABLE IF NOT EXISTS detection_history
             (id INTEGER PRIMARY KEY AUTOINCREMENT, class_name text, last_visit_time text)''')

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Load the YOLOv5m model trained for people detection

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'rtsp://<ip_address>' for IP camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy of the frame
    frame_copy = frame.copy()

    # Get the detections
    results = model(frame)

    # Loop through the detections
    for result in results:
        # Check if any objects were detected
        if len(result.boxes.cls) > 0:
            # Get the class name and confidence
            class_name = "person"  # Only detect people
            confidence = result.boxes.conf[0].item()

            # Get the bounding box coordinates
            x_min, y_min, x_max, y_max = result.boxes.xyxy[0].numpy()

            # Draw the bounding box
            cv2.rectangle(frame_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Draw the class name and confidence
            cv2.putText(frame_copy, f"{class_name} {confidence:.2f}", (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store the detection history
            c.execute("INSERT INTO detection_history (class_name, last_visit_time) VALUES (?,?)",
                      (class_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    # Display the output
    cv2.imshow('Object Detection', frame_copy)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
conn.close()