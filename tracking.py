import os
import cv2
import numpy as np
import time
import torch
import torchreid
import torchvision.transforms as t
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Define YOLOv8 class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", 
    "toothbrush"
]

# Function to load pretrained yolo model
def load_yolo_model():
    # loading pretrained model
    yolo_model = YOLO('yolov8n.pt')
    return yolo_model

# Function to load pretrained reid model
def load_reid_model():
    reid_model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )
    reid_model.eval()
    return reid_model

# Function to extract feature for re-identification
def extract_features(reid_model, cropped_img):
    transform = t.Compose([
        t.Resize((256, 128)),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(cropped_img)
    img_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        features = reid_model(img_tensor)
    return features.numpy().flatten()

# Implementing tracking algorithm
def process_each_frame(frame, yolo_model, reid_model):
    features_list = []
    bbs = []

    # Initialize tracker 
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    # Initialize a dictionary to keep track of counts for each class
    class_count = {}
    track_id_class_map = {}  # Maps track IDs to their class names for unique counting

    results = yolo_model(frame)

    # Set a confidence threshold
    confidence_threshold = 0.5

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0]
            if conf < confidence_threshold:
                continue  # Skip low-confidence detections

            x1, y1, x2, y2 = box.xyxy[0]
            cls = int(box.cls[0])

            # Debugging: Print detected class index and name
            print(f"Detected class index: {cls}, name: {CLASS_NAMES[cls]}")

            # Convert to integer pixel values
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cropped_img = frame[y1:y2, x1:x2]

            width = x2 - x1
            height = y2 - y1

            bb = ([x1, y1, width, height], conf, cls)
            bbs.append(bb)

            # Extract re-ID features
            features = extract_features(reid_model, cropped_img)
            features_list.append(features)

    # Update tracker with detections and re-ID features
    tracked_objects = tracker.update_tracks(bbs, [conf for _, conf, _ in bbs], [cls for _, _, cls in bbs], frame, features_list)

    for obj in tracked_objects:
        bbox = obj.to_tlbr()
        track_id = obj.track_id
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        
        # Get the class index and name
        class_index = int(bbs[tracked_objects.index(obj)][2])  # Use the tracked object index to get the class index
        class_name = CLASS_NAMES[class_index]  # Get the class name using the class index

        # Only update the count if this track ID is new for the class
        if track_id not in track_id_class_map:
            track_id_class_map[track_id] = class_name  # Map track ID to class name
            if class_name in class_count:
                class_count[class_name] += 1  # Increment unique count
            else:
                class_count[class_name] = 1  # Initialize count

        # Draw bounding box and class name with count
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {class_count[class_name]}: {track_id}", 
                    (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# function to perform object identification - feature extraction - tracking for the input video source
def mot_with_reid(video_source=0):
    
    yolo_model = load_yolo_model()
    reid_model = load_reid_model()

    if yolo_model is None or reid_model is None:
        print("Error in loading the model")
        return
    # Capture video from the webcam
    cap = cv2.VideoCapture(video_source)

    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Run for a total of 2 minutes, alternating modes
    total_duration = 120  # Total duration in seconds
    start_time = time.time()
    mode_duration = 60  # Duration for each mode (1 minute)
    
    while True:
        
        if (int(time.time() - start_time) // mode_duration) % 2 == 0:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break

            # Pass the models directly into the function
            processed_frame = process_each_frame(frame, yolo_model, reid_model)

            cv2.imshow('Live Object Detection - One Frame', processed_frame)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                print("Interrupted by user.")
                break

        else:
            for _ in range(2):
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame")
                    break

                processed_frame = process_each_frame(frame, yolo_model, reid_model)
                cv2.imshow('Live Object Detection - Live Video', processed_frame)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Interrupted by user.")
                    break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    while True:
        print("\nSelect an option:")
        print("1. Video File Detection")
        print("2. Live Video Detection")
        print("3. Exit")
        choice = input("Enter your choice (1-4): ")

        """
        if choice == '1':
            input_directory = input("Enter the path to the image directory: ")
            if os.path.isdir(input_directory):
                mot_with_reid(input_directory)
            else:
                print("Invalid directory. Please try again.")
        """
        if choice == '1':
            print("\n Enter the path to the video file: ")
            video_file = input().strip()
            if os.path.isfile(video_file):
                mot_with_reid(video_file)
            else:
                print("Invalid file path. Please try again.")
        
        elif choice == '2':
            mot_with_reid()
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
