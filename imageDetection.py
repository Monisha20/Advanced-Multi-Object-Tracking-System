import os
import cv2
from ultralytics import YOLO

# Function to detect objects
def detect_objects(input_directory):
    model_identify = YOLO('yolov8n.pt') # Load the YOLOv8 model
    model_segment = YOLO('yolov8n-seg.pt')

    # Loop through each image in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for valid image file extensions
            image_path = os.path.join(input_directory, filename)
            print(f"Trying to read image from: {image_path}")  # Debug statement
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")  # Check if image is loaded successfully
                continue
            
            # Perform object detection
            results = model_identify(img)

            # Loop through each detected object
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                result.show()  # display to screen
                result.save(filename="result.jpg")  # save to disk


#function to perform real time object detection
def detect_objects_in_video(video_source=0):
    # Load the YOLOv8 model
    model_identify = YOLO('yolov8n.pt')
    
    # Open the video source
    cap = cv2.VideoCapture(video_source)
    
    # Check if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
            
        # Perform object detection
        results = model_identify(frame)

        # Loop through the results and draw bounding boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract the coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates
                conf = box.conf[0]  # Get the confidence score
                cls = int(box.cls[0])  # Get the class index

                # Draw a rectangle and label on the frame
                label = f'{model_identify.names[cls]}: {conf:.2f}'  # Class name and confidence
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw rectangle
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Draw label

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)
        
        # Check for 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_directory = "C:\\Users\\bindu\\Documents\\Monisha\\ML_Project\\Workspaces\\MOT_Coco_Data_Cleanup\\coco-dataset_local\\val2017\\val2017"  # Path to the directory containing images
    input_video_directory = "C:\\Users\\bindu\\Documents\\Monisha\\ML_Project\\Workspaces\\MOT_Coco_Data_Cleanup\\coco-dataset_local\\fruit-and-vegetable-detection.mp4"
    detect_objects(input_directory)
    #detect_objects_in_video()