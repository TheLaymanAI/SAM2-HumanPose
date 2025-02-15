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

def process_video(video_path, output_path="output_video.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to read the video.")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Double width for side-by-side comparison
    
    # Get user input for transparency level
    alpha = float(input("Enter transparency percentage for the mask (0-100): ")) / 100
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw keypoints on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extract keypoint coordinates
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
            segmentation_results = model(frame, bboxes=bboxes)
            mask = segmentation_results[0].masks.data[0].cpu().numpy()  # First detected mask
            
            # Apply red mask with user-defined transparency
            red_overlay = np.zeros_like(frame, dtype=np.uint8)
            red_overlay[:, :, 1] = 255  # Red channel
            segmented_frame = frame.copy()
            segmented_frame[mask > 0] = cv2.addWeighted(frame[mask > 0], 1 - alpha, red_overlay[mask > 0], alpha, 0)
        else:
            segmented_frame = frame.copy()
        
        # Combine original and processed frames side by side
        comparison_frame = np.hstack((original_frame, segmented_frame))
        
        # Add labels
        cv2.putText(comparison_frame, "Original Video", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_frame, "SAM2 + Google MediaPipe", (width + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Comparison', comparison_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        out.write(comparison_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved at {output_path}")

# Example usage
input_transparency = input("Enter transparency percentage for the mask (0-100): ")
input_video = input("Enter input video path : ")
output_video = input("Enter output video path :")

process_video(input_video, input_transparency , output_video)
