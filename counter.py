import cv2
import torch
import yolov5 
from sort import *
import numpy as np


model_weight = "./models/yolov5s.pt"
#load YOLOv5 model
model = torch.hub.load('./yolov5', 'custom', path=model_weight, source='local')

#load the video
video_path='./media/video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer (optional)
output_path = "output_video.mp4"
out = cv2.VideoWriter(
    output_path, 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (frame_width, frame_height)
)

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Initialize counters
total_cars = 0
tracked_ids = set()  # To avoid double-counting unique IDs

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Assuming results.xyxy[0] contains the detections

    # Ensure detections have the correct format
    sort_detections = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det[:6]
        sort_detections.append([x1, y1, x2, y2, score, class_id])

    sort_detections = np.array(sort_detections)

    # Update tracker
    tracked_objects = tracker.update(sort_detections)

    for obj in tracked_objects:
        if len(obj) >= 5:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            label = f"Car ID {track_id}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.rectangle(frame, (10, 45), (150, 70), (0, 0, 0), -1)  # Black rectangle to clear the area
            cv2.putText(frame, f"new ID: {track_id}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Process the tracked object
            # Process the tracked object
            if track_id not in tracked_ids:
                tracked_ids.add(track_id)
                print(f"New car detected: {track_id}")       
        # Update and display the total count of cars
            cv2.putText(frame, f"Total cars: {total_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print(f"Unexpected object format: {obj}")

        # Count unique car IDs


    # print total cars detected
    # Show frame (optional)
    #cv2.imshow("Traffic Video", frame)

    # Write to output video
    out.write(frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
total_cars = len(tracked_ids)
print(total_cars)
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total cars detected: {total_cars}")