import cv2
import time
from ultralytics import YOLO

def detect_objects_in_live_video(video_source=0):
    # Load the pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

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
        elapsed_time = time.time() - start_time

        # Check if the total duration has been reached
        if elapsed_time >= total_duration:
            print("Total duration reached.")
            break

        # Mode 1: One frame detection for 1 minute
        if (elapsed_time // mode_duration) % 2 == 0:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame")
                break
            
            # Perform object detection on the frame
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
            cv2.imshow('Live Object Detection - One Frame', frame)
            key = cv2.waitKey(1000)  # Show the frame for 1 second before moving on

            # Check for keyboard interruption
            if key & 0xFF == ord('q'):
                print("Interrupted by user.")
                break

        # Mode 2: Two frames live video for 1 minute
        else:
            for _ in range(2):  # Show 2 frames
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame")
                    break
                
                # Display the live video frame
                cv2.imshow('Live Object Detection - Live Video', frame)

                # Check for keyboard interruption
                key = cv2.waitKey(200)  # Show each frame for 0.2 seconds
                if key & 0xFF == ord('q'):
                    print("Interrupted by user.")
                    break

            # Wait for the remainder of the time to fill the 1-minute slot
            time.sleep(30)  # Adjust this if necessary

    # Release resources upon quitting
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects_in_live_video()  # Run the live detection
