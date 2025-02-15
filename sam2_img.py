import cv2
import numpy as np
import mediapipe as mp
from ultralytics import SAM

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load SAM model
model = SAM("sam2_b.pt")

def detect_pose_and_segment(image_path, save_path="output.png"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to read the image.")
    
    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_image)
    
    if not results.pose_landmarks:
        raise ValueError("No human pose detected.")
    
    # Draw keypoints on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Extract keypoint coordinates
    height, width, _ = image.shape
    x_min, y_min, x_max, y_max = width, height, 0, 0
    
    for landmark in results.pose_landmarks.landmark:
        x, y = int(landmark.x * width), int(landmark.y * height)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    
    # Expand the bounding box slightly
    padding = 20
    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(width, x_max + padding), min(height, y_max + padding)
    
    # Bounding box
    bboxes = [[x_min, y_min, x_max, y_max]]
    
    # Run segmentation
    segmentation_results = model(image_path, bboxes=bboxes)
    mask = segmentation_results[0].masks.data[0].cpu().numpy()  # First detected mask
    
    # Create blue segmentation overlay
    blue_mask = np.zeros_like(image)
    blue_mask[:, :, 0] = mask * 255  # Blue channel
    overlay = cv2.addWeighted(image, 0.7, blue_mask, 0.3, 0)
    
    # Save output
    cv2.imwrite(save_path, overlay)
    print(f"Segmented image saved at {save_path}")


input_img = input("Enter input video path : ")
output_img = input("Enter output video path :")

detect_pose_and_segment(input_img , output_img)
