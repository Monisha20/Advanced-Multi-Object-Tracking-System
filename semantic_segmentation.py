import os
import time
import cv2
import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Function to decode the segmentation map
def decode_segmentation(segmentation_map):
    LABEL_COLORS = np.random.randint(0, 255, size=(21, 3), dtype="uint8")
    r = LABEL_COLORS[segmentation_map, 0]
    g = LABEL_COLORS[segmentation_map, 1]
    b = LABEL_COLORS[segmentation_map, 2]
    rgb_image = np.stack([r, g, b], axis=2)
    return rgb_image

def add_labels_to_frame(segmentation_map, frame, original_frame_shape):

    CLASS_NAMES =[
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
    unique_classes = np.unique(segmentation_map)  # Find all the classes present in the frame
    scale_x = original_frame_shape[1] / segmentation_map.shape[1]
    scale_y = original_frame_shape[0] / segmentation_map.shape[0]

    for cls in unique_classes:
        if cls == 0:
            continue
        mask = (segmentation_map == cls)
        
        y, x = np.where(mask)
        if len(x) > 0 and len(y) > 0:
            centroid_x, centroid_y = np.mean(x), np.mean(y)
            label = CLASS_NAMES[cls]
            
            label_x = int(centroid_x * scale_x)
            label_y = int(centroid_y * scale_y)
            
            cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def process_image_for_segmentation_and_detection(image_file):
    # Load the image
    image = cv2.imread(image_file)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    processed_image = process_each_frame(image)
    
    # Show the processed image
    cv2.imshow('Image Segmentation and Object Detection', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_each_frame(frame):

    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    preprocess = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per model requirements
    ])

    original_frame_shape = frame.shape

    resized_frame = cv2.resize(frame, (520, 520))

    input_tensor = preprocess(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).numpy()

    decoded_segmentation = decode_segmentation(output_predictions)

    decoded_segmentation_resized = cv2.resize(decoded_segmentation, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = output_predictions != 0

    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    frame[mask_resized == 1] = decoded_segmentation_resized[mask_resized == 1]  # Overlay only on the detected objects

    labeled_frame = add_labels_to_frame(output_predictions, frame, original_frame_shape)

    return labeled_frame

def capture_video(video_source=0):

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

            processed_frame = process_each_frame(frame)

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

                processed_frame = process_each_frame(frame)
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
        print("3. Image Detection")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            print("\n Enter the path to the video file: ")
            video_file = input().strip()
            if os.path.isfile(video_file):
                capture_video(video_file)
            else:
                print("Invalid file path. Please try again.")
        
        elif choice == '2':
            capture_video()
        
        elif choice == '3':
            print("\n Enter the path to the image file: ")
            image_file = input().strip()
            if os.path.isfile(image_file):
                process_image_for_segmentation_and_detection(image_file)
            else:
                print("Invalid file path. Please try again.")

        elif choice == '4':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()