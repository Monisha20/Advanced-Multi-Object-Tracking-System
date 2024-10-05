import os
import cv2
from ultralytics import YOLO

#function to detect the objects in images
def detect_objects_in_images(input_directory):
    #Loading the pretrained yolov8n model
    model = YOLO('yolov8n.pt')

    output_dir = 'object_identification_output/'
    os.makedirs(output_dir,exist_ok=True)

    #Iterates through all the images in the input image directory
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_directory, filename)
            print(f"Trying to read image from: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to read image: {image_path}")
                continue

            #performing object detection
            results = model(img)

            for result in results:
                result.show() #display the detected image
                output_filename = os.path.join(output_dir, f"detected_{filename}")
                output_image = result.plot()
                cv2.imwrite(output_filename,output_image)

#function to detect the objects in video
#parameters: if no arguments received - reads input from webcam, else accepts path to video.mp4 file
def detect_objects_in_video(video_source=0):

    #Loading the pretrained yolov8n model
    model = YOLO('yolov8n.pt')
    
    cap = cv2.VideoCapture(video_source)
    
    # Checks if the video source was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break
            
        #performing object detection on each frame
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  #coordinates
                conf = box.conf[0]  #confidence score
                cls = int(box.cls[0])  #class index

                # Drawing a rectangle and label on the frame
                label = f'{model.names[cls]}: {conf:.2f}'
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)
        
        # Check for 'q' - key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
    # Release the resource upon quit
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #input_dir: holds the path to images directory
    input_dir = "C:\\Users\\bindu\\Documents\\Monisha\\ML_Project\\Workspaces\\MOT_Coco_Data_Cleanup\\coco-dataset_local\\val2017\\val2017"
    #input_video_dir: holds the path to video file
    input_video_dir = "C:\\Users\\bindu\\Documents\\Monisha\\ML_Project\\Workspaces\\MOT_Coco_Data_Cleanup\\coco-dataset_local\\fruit-and-vegetable-detection.mp4"

    detect_objects_in_images(input_dir)
    #detect_objects_in_video()