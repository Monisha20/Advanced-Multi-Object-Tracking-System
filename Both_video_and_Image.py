import os
import cv2
from ultralytics import YOLO

# Function to detect objects in images
def detect_objects_in_images(input_directory):
    model = YOLO('yolov8n.pt')
    output_dir = 'object_identification_output/'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            print(f"Trying to read image from: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Performing object detection
            results = model(img)

            for result in results:
                output_filename = os.path.join(output_dir, f"detected_{filename}")
                output_image = result.plot()
                cv2.imwrite(output_filename, output_image)
                result.show()  # Display the detected image

# Function to detect objects in video
def detect_objects_in_video(video_source):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
            
        # Performing object detection on each frame
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class index

                # Draw a rectangle and label on the frame
                label = f'{model.names[cls]}: {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    # Release resources upon quitting
    cap.release()
    cv2.destroyAllWindows()

# Function to handle live video detection
def detect_objects_in_live_video():
    print("Starting live video detection...")
    detect_objects_in_video(0)  # 0 for default webcam

# Main function to present options to the user
def main():
    while True:
        print("\nSelect an option:")
        print("1. Image Detection")
        print("2. Video File Detection")
        print("3. Live Video Detection")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            input_directory = input("Enter the path to the image directory: ")
            if os.path.isdir(input_directory):
                detect_objects_in_images(input_directory)
            else:
                print("Invalid directory. Please try again.")
        
        elif choice == '2':
            video_file = input("Enter the path to the video file: ")
            if os.path.isfile(video_file):
                detect_objects_in_video(video_file)
            else:
                print("Invalid file path. Please try again.")
        
        elif choice == '3':
            detect_objects_in_live_video()
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()
