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

#Function to load pretrained yolo model
def load_yolo_model():
    #loading pretrained model
    yolo_model = YOLO('yolov8n.pt')
    return yolo_model

#Function to load pretrained reid model
def load_reid_model():
    reid_model = torchreid.models.build_model(
        name='resnet50',
        num_classes=1000,
        pretrained=True
    )
    reid_model.eval()
    return reid_model

#Fucntion to extract feature for re identification
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

#Implementing tracking algorithm
def process_each_frame(frame, previous_features):

    features_list = []
    bbs = []

    yolo_model = load_yolo_model()
    reid_model = load_reid_model()

    if yolo_model is None or reid_model is None:
        print("Error in loading the model")
        return

    # YOLOv8 class names
    class_names = [
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

    #initialize tracker 
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

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

    for obj in tracked_objects:
        bbox = obj.to_tlbr()
        track_id = int(obj.track_id)
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        class_index = int(bbs[tracked_objects.index(obj)][2])
        class_name = class_names[class_index]

        if track_id < len(features_list):
            current_features = features_list[track_id]
            previous_features[track_id] = current_features

        draw_bounding_box(frame, top_left, bottom_right, class_name, track_id)

    return frame, previous_features


def draw_bounding_box(frame, top_left, bottom_right, class_name, track_id):
    # Create the label
    label = f"{class_name} ID: {track_id}"

    # Draw the rectangle around the detected object
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Calculate the position for the label
    text_position = (top_left[0], top_left[1] - 10)

    # Draw the label with a different background for better visibility
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(frame, (top_left[0], top_left[1] - text_size[1] - 5),
                  (top_left[0] + text_size[0], top_left[1]), (0, 255, 0), -1)  # Filled rectangle for label background
    cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

#function to perform object identification - feature extraction - tracking for the input video source
def mot_with_reid(video_source=0):
    
    previous_features = {}

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

            processed_frame, previous_features = process_each_frame(frame, previous_features)

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

                processed_frame, previous_features = process_each_frame(frame, previous_features)
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