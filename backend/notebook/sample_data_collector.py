#!/usr/bin/env python3
"""
Sample Data Collection Script
============================

This script helps you quickly collect sample data for common sign language gestures.
It provides predefined signs and guided collection process.

Usage:
    python sample_data_collector.py
"""

import sys
import os
import time
from pathlib import Path

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_collector import SignLanguageDataCollector

# Predefined common signs
COMMON_SIGNS = [
    "HELLO",
    "GOODBYE", 
    "THANK_YOU",
    "PLEASE",
    "YES",
    "NO",
    "SORRY",
    "EXCUSE_ME",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE"
]

def collect_sample_data():
    """Collect sample data for predefined signs"""
    print("🎯 Sample Data Collection for Sign Language")
    print("=" * 50)
    print("This will help you collect training data for common signs.")
    print("Each sign will be recorded multiple times for better training.")
    print("=" * 50)
    
    # Create data collector
    collector = SignLanguageDataCollector()
    
    print("\n📋 Predefined signs to collect:")
    for i, sign in enumerate(COMMON_SIGNS, 1):
        print(f"   {i:2d}. {sign}")
    
    print(f"\n🎬 Collection Plan:")
    print(f"   - {len(COMMON_SIGNS)} signs")
    print(f"   - 10 sequences per sign")
    print(f"   - 30 frames per sequence")
    print(f"   - Total: {len(COMMON_SIGNS) * 10} sequences")
    
    # Ask user which signs to collect
    print(f"\nSelect signs to collect:")
    print("   'all' - Collect all signs")
    print("   '1,2,3' - Collect specific signs by number")
    print("   'custom' - Enter custom sign names")
    
    choice = input("Your choice: ").strip().lower()
    
    if choice == 'all':
        signs_to_collect = COMMON_SIGNS
    elif choice == 'custom':
        custom_signs = input("Enter sign names (comma-separated): ").strip().upper()
        signs_to_collect = [s.strip() for s in custom_signs.split(',') if s.strip()]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            signs_to_collect = [COMMON_SIGNS[i] for i in indices if 0 <= i < len(COMMON_SIGNS)]
        except (ValueError, IndexError):
            print("❌ Invalid choice. Using all signs.")
            signs_to_collect = COMMON_SIGNS
    
    if not signs_to_collect:
        print("❌ No signs selected. Exiting.")
        return
    
    print(f"\n✅ Will collect data for: {', '.join(signs_to_collect)}")
    
    # Confirm collection
    response = input("\nStart data collection? (Y/n): ").lower()
    if response == 'n':
        print("Data collection cancelled.")
        return
    
    # Start collection
    print(f"\n🚀 Starting data collection...")
    print("Instructions:")
    print("1. Press SPACE to start recording")
    print("2. Perform the sign gesture for ~3 seconds")
    print("3. Press SPACE to stop recording")
    print("4. Repeat 10 times for each sign")
    print("5. Press Q to quit early")
    print("=" * 50)
    
    # Initialize webcam
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    recording = False
    current_sign_idx = 0
    current_sequence_count = 0
    sequences_per_sign = 10
    
    while current_sign_idx < len(signs_to_collect):
        current_sign = signs_to_collect[current_sign_idx]
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add instructions to frame
        cv2.putText(frame, f"Sign: {current_sign}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Progress: {current_sign_idx + 1}/{len(signs_to_collect)}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Sequence: {current_sequence_count + 1}/{sequences_per_sign}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Recording: {'YES' if recording else 'NO'}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if recording else (255, 255, 255), 2)
        cv2.putText(frame, "SPACE: Start/Stop | Q: Quit", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Sample Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            if not recording:
                # Start recording
                recording = True
                collector.current_sign = current_sign
                collector.current_sequence = []
                print(f"🎬 Recording {current_sign} - Sequence {current_sequence_count + 1}")
            else:
                # Stop recording
                recording = False
                if len(collector.current_sequence) >= collector.sequence_length:
                    collector.save_sequence()
                    current_sequence_count += 1
                    print(f"✅ Saved {current_sign} - Sequence {current_sequence_count}")
                    
                    # Move to next sign if complete
                    if current_sequence_count >= sequences_per_sign:
                        current_sign_idx += 1
                        current_sequence_count = 0
                        print(f"🎉 Completed {current_sign}! Moving to next sign...")
                else:
                    print(f"⚠️  Sequence too short ({len(collector.current_sequence)} frames)")
                collector.current_sequence = []
        
        # Record frames if recording
        if recording:
            # Convert frame to bytes for pose estimation
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Extract keypoints
            keypoints = collector.pose_estimator.extract_keypoints(frame_bytes)
            if keypoints is not None:
                collector.current_sequence.append(keypoints)
                
                # Check if sequence is complete
                if len(collector.current_sequence) >= collector.sequence_length:
                    recording = False
                    collector.save_sequence()
                    current_sequence_count += 1
                    print(f"✅ Saved {current_sign} - Sequence {current_sequence_count}")
                    
                    # Move to next sign if complete
                    if current_sequence_count >= sequences_per_sign:
                        current_sign_idx += 1
                        current_sequence_count = 0
                        print(f"🎉 Completed {current_sign}! Moving to next sign...")
                    collector.current_sequence = []
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    stats = collector.get_collection_stats()
    print("\n📊 Collection Statistics:")
    for sign, count in stats.items():
        print(f"   {sign}: {count} sequences")
    
    total_sequences = sum(stats.values())
    print(f"\n🎉 Data collection completed!")
    print(f"   Total sequences collected: {total_sequences}")
    print(f"   Signs collected: {len(stats)}")
    
    if total_sequences > 0:
        print(f"\n🚀 Ready to train model!")
        print(f"   Run: python quick_train.py")

def main():
    """Main function"""
    collect_sample_data()

if __name__ == "__main__":
    main() 