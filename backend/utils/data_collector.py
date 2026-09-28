import cv2
import numpy as np
import os
import json
import time
from pose_estimation import PoseEstimator
from typing import Dict, List, Tuple

class SignLanguageDataCollector:
    """Collect sign language training data from webcam"""
    
    def __init__(self, data_dir="data/raw", sequence_length=30):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.pose_estimator = PoseEstimator(sequence_length=sequence_length)
        self.current_sign = None
        self.current_sequence = []
        self.collected_data = {}
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
    def start_collection(self):
        """Start the data collection process"""
        print("🎯 Sign Language Data Collection System")
        print("=" * 50)
        print("Instructions:")
        print("1. Enter the sign name (e.g., 'HELLO', 'THANK_YOU')")
        print("2. Press 'SPACE' to start recording")
        print("3. Perform the sign gesture")
        print("4. Press 'SPACE' to stop recording")
        print("5. Press 'Q' to quit")
        print("=" * 50)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
            
        recording = False
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add instructions to frame
            cv2.putText(frame, f"Sign: {self.current_sign or 'None'}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Recording: {'YES' if recording else 'NO'}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {len(self.current_sequence)}/{self.sequence_length}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE: Start/Stop Recording | Q: Quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Sign Language Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                if not recording:
                    # Start recording
                    if not self.current_sign:
                        self.current_sign = input("Enter sign name: ").strip().upper()
                        if not self.current_sign:
                            continue
                    
                    recording = True
                    self.current_sequence = []
                    frame_count = 0
                    print(f"🎬 Started recording for sign: {self.current_sign}")
                else:
                    # Stop recording
                    recording = False
                    if len(self.current_sequence) >= self.sequence_length:
                        self.save_sequence()
                    else:
                        print(f"⚠️  Sequence too short ({len(self.current_sequence)} frames). Need {self.sequence_length} frames.")
                    self.current_sequence = []
                    frame_count = 0
            
            # Record frames if recording
            if recording:
                # Convert frame to bytes for pose estimation
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Extract keypoints
                keypoints = self.pose_estimator.extract_keypoints(frame_bytes)
                if keypoints is not None:
                    self.current_sequence.append(keypoints)
                    frame_count += 1
                    
                    # Check if sequence is complete
                    if len(self.current_sequence) >= self.sequence_length:
                        recording = False
                        self.save_sequence()
                        self.current_sequence = []
                        frame_count = 0
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Data collection completed!")
        
    def save_sequence(self):
        """Save the collected sequence"""
        if len(self.current_sequence) < self.sequence_length:
            return
            
        # Create sign directory
        sign_dir = os.path.join(self.data_dir, self.current_sign)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"{self.current_sign}_{timestamp}.npy"
        filepath = os.path.join(sign_dir, filename)
        
        # Save sequence as numpy array
        sequence_array = np.array(self.current_sequence)
        np.save(filepath, sequence_array)
        
        print(f"💾 Saved sequence: {filepath}")
        print(f"   Shape: {sequence_array.shape}")
        print(f"   Sign: {self.current_sign}")
        
        # Update collected data stats
        if self.current_sign not in self.collected_data:
            self.collected_data[self.current_sign] = 0
        self.collected_data[self.current_sign] += 1
        
    def get_collection_stats(self):
        """Get statistics about collected data"""
        stats = {}
        for sign in os.listdir(self.data_dir):
            sign_dir = os.path.join(self.data_dir, sign)
            if os.path.isdir(sign_dir):
                files = [f for f in os.listdir(sign_dir) if f.endswith('.npy')]
                stats[sign] = len(files)
        return stats

def main():
    """Main function to run data collection"""
    collector = SignLanguageDataCollector()
    collector.start_collection()
    
    # Print final statistics
    stats = collector.get_collection_stats()
    print("\n📊 Collection Statistics:")
    for sign, count in stats.items():
        print(f"   {sign}: {count} sequences")

if __name__ == "__main__":
    main() 