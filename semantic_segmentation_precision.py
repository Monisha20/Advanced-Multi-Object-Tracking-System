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

# Function to load pretrained yolo model
def load_yolo_model():
    yolo_model = YOLO('yolov8n.pt')
    return yolo_model

# Function to load pretrained reid model
def load_reid_model():
    reid_model = torchreid.models.build_model(
        name='resnet50',
        num_classes=1000,
        pretrained=True
    )
    reid_model.eval()
    return reid_model

# Function to extract features for re-identification
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
def process_each_frame(frame, previous_features, tracker):
    features_list = []
    bbs = []

    yolo_model = load_yolo_model()
    reid_model = load_reid_model()

    if yolo_model is None or reid_model is None:
        print("Error in loading the model")
        return

    # YOLOv8 class names
    class_names = [...]  # Keep your existing class names here

    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            x1 = int(np.round(x1))
            y1 = int(np.round(y1))
            x2 = int(np.round(x2))
            y2 = int(np.round(y2))

            cropped_img = frame[y1:y2, x1:x2]
            width = x2 - x1
            height = y2 - y1
            bb = ([x1, y1, width, height], conf, cls)
            bbs.append(bb)

            # Extract re-ID features
            features = extract_features(reid_model, cropped_img)
            features_list.append(features)

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(bbs, [conf for _, conf, _ in bbs], [cls for _, _, cls in bbs], frame, features_list)

    return tracked_objects, previous_features

def draw_bounding_box(frame, top_left, bottom_right, class_name, track_id):
    # Create the label
    label = f"{class_name} ID: {track_id}"

    # Draw the rectangle around the detected object
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    text_position = (top_left[0], top_left[1] - 10)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (top_left[0], top_left[1] - text_size[1] - 5),
                  (top_left[0] + text_size[0], top_left[1]), (0, 255, 0), -1)
    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def evaluate_tracking_performance(tracked_objects, ground_truth):
    """
    Evaluate tracking performance based on precision, recall, and F1 score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    tracked_ids = set(obj[0] for obj in tracked_objects)
    ground_truth_ids = set(gt[0] for gt in ground_truth)

    for track_id in tracked_ids:
        if track_id in ground_truth_ids:
            true_positives += 1
        else:
            false_positives += 1

    for track_id in ground_truth_ids:
        if track_id not in tracked_ids:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# Function to perform object identification - feature extraction - tracking for the input video source
def mot_with_reid(video_source=0, ground_truth=[]):
    previous_features = {}
    tracked_objects = []  # To store tracked objects for evaluation
    ground_truth = ground_truth  # Update with your ground truth data

    # Capture video from the webcam
    cap = cv2.VideoCapture(video_source)

    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Initialize tracker
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

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

            tracked_objects_current, previous_features = process_each_frame(frame, previous_features, tracker)
            tracked_objects.extend(tracked_objects_current)  # Append current tracked objects
            
            cv2.imshow('Live Object Detection - One Frame', frame)
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

                tracked_objects_current, previous_features = process_each_frame(frame, previous_features, tracker)
                tracked_objects.extend(tracked_objects_current)  # Append current tracked objects
                
                cv2.imshow('Live Object Detection - Live Video', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Interrupted by user.")
                    break

    cap.release()
    cv2.destroyAllWindows()

    # Evaluate performance after video processing
    performance_metrics = evaluate_tracking_performance(tracked_objects, ground_truth)
    print("Precision:", performance_metrics["precision"])
    print("Recall:", performance_metrics["recall"])
    print("F1 Score:", performance_metrics["f1_score"])

# Main function
def main():
    while True:
        print("\nSelect an option:")
        print("1. Video File Detection")
        print("2. Live Video Detection")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            print("\nEnter the path to the video file: ")
            video_file = input().strip()
            if os.path.isfile(video_file):
                # Provide the ground truth for evaluation (should be prepared separately)
                ground_truth = []  # Populate this with your actual ground truth data
                mot_with_reid(video_file, ground_truth)
            else:
                print("Invalid file path. Please try again.")
        
        elif choice == '2':
            ground_truth = []  # Populate this with your actual ground truth data
            mot_with_reid(0, ground_truth)
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
