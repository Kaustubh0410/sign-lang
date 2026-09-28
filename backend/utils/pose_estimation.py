import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import torch

class PoseEstimator:
    def __init__(self, sequence_length=30):
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_length = sequence_length
        self.keypoint_sequence = deque(maxlen=sequence_length)
        
    def extract_keypoints(self, frame_bytes: bytes):
        """Extract keypoints from a single frame"""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print("Failed to decode image")
                return np.zeros(153)  # Return zeros if image decode fails
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model.process(img_rgb)
            
            # Extract landmarks with confidence scores
            landmarks = []
            
            # Left hand landmarks (21 points × 3 coordinates)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * 21)  # Pad with zeros
                
            # Right hand landmarks (21 points × 3 coordinates)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * 21)  # Pad with zeros
                
            # Pose landmarks (33 points × 3 coordinates) - focus on upper body
            relevant_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Key upper body points
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                # Select relevant pose points (shoulders, arms, head)
                for idx in relevant_indices:
                    lm = pose_landmarks[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * len(relevant_indices))
                
            keypoints = np.array(landmarks)
            print(f"Extracted keypoints: {keypoints.shape}, sum: {np.sum(keypoints)}")
            return keypoints
            
        except Exception as e:
            print(f"Error in extract_keypoints: {e}")
            return np.zeros(153)  # Return zeros on error
    
    def get_sequence_for_prediction(self, frame_bytes: bytes):
        """Get temporal sequence for gesture recognition"""
        try:
            keypoints = self.extract_keypoints(frame_bytes)
            self.keypoint_sequence.append(keypoints)
            
            print(f"Sequence length: {len(self.keypoint_sequence)}/{self.sequence_length}")
            
            # Return sequence if we have enough frames
            if len(self.keypoint_sequence) == self.sequence_length:
                sequence = np.array(list(self.keypoint_sequence))
                print(f"Full sequence shape: {sequence.shape}")
                return sequence
            return None
            
        except Exception as e:
            print(f"Error in get_sequence_for_prediction: {e}")
            return None
    
    def reset_sequence(self):
        """Reset the temporal sequence"""
        self.keypoint_sequence.clear()
        print("Sequence reset")