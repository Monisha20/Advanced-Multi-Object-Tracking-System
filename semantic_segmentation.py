import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Load the pre-trained DeepLabV3 model from torchvision
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define the image transformation (resize, normalize, etc.)
preprocess = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image (numpy array) to PIL image
    transforms.Resize((520, 520)),  # Resize image to fit the model input size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per model requirements
])

# Color mapping for segmentation classes
LABEL_COLORS = np.random.randint(0, 255, size=(21, 3), dtype="uint8")

# Function to decode the segmentation map
def decode_segmentation(segmentation_map):
    r = LABEL_COLORS[segmentation_map, 0]
    g = LABEL_COLORS[segmentation_map, 1]
    b = LABEL_COLORS[segmentation_map, 2]
    rgb_image = np.stack([r, g, b], axis=2)
    return rgb_image

# Open the input video using OpenCV
video_path = "C:\\Users\\bindu\\Documents\\Monisha\\ML_Project\\Workspaces\\object_tracking_workspace\\input\\person-bicycle-car-detection.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
out = cv2.VideoWriter('output_segmented_video.mp4', fourcc, 30.0, (520, 520))  # Output video

# Process the video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension

    # Perform inference (disable gradient calculation)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Get the predicted class for each pixel
    output_predictions = output.argmax(0).cpu().numpy()

    # Decode the segmentation result into an RGB image
    decoded_segmentation = decode_segmentation(output_predictions)

    # Convert the decoded segmentation to uint8 (needed for writing to video)
    decoded_segmentation = decoded_segmentation.astype(np.uint8)

    # Write the frame with segmentation into the output video
    out.write(decoded_segmentation)

    # (Optional) Display the current frame with segmentation
    cv2.imshow('Segmented Frame', decoded_segmentation)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release the video reader and writer
cap.release()
out.release()
cv2.destroyAllWindows()