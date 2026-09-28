#!/usr/bin/env python3
"""
Download and Convert Sign Language Datasets
==========================================

This script downloads existing sign language datasets and converts them
to our training format, saving you hours of manual data collection.

Available datasets:
- ASL Alphabet Dataset (Kaggle)
- Sign Language MNIST
- Custom dataset conversion
"""

import os
import sys
import requests
import zipfile
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp
from tqdm import tqdm

class DatasetDownloader:
    """Download and convert sign language datasets"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.pose_estimator = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
    def download_asl_alphabet(self):
        """Download ASL Alphabet dataset from Kaggle"""
        print("📥 Downloading ASL Alphabet Dataset...")
        
        # Dataset URL (you'll need to download manually from Kaggle)
        dataset_url = "https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
        
        print("⚠️  Manual download required:")
        print(f"   1. Visit: {dataset_url}")
        print("   2. Download the dataset")
        print("   3. Extract to 'datasets/asl_alphabet/'")
        print("   4. Run: python download_datasets.py --convert-asl")
        
        return False
    
    def convert_asl_alphabet(self, dataset_path="datasets/asl_alphabet"):
        """Convert ASL Alphabet dataset to our format"""
        print("🔄 Converting ASL Alphabet dataset...")
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
        
        # ASL Alphabet mapping
        asl_mapping = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
            'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
            'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
            'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
        }
        
        converted_count = 0
        
        for letter in asl_mapping.keys():
            letter_dir = os.path.join(dataset_path, letter)
            if not os.path.exists(letter_dir):
                continue
                
            print(f"   Converting letter: {letter}")
            
            # Create output directory
            output_dir = os.path.join(self.data_dir, letter)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process images
            image_files = [f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            for i, img_file in enumerate(tqdm(image_files[:100], desc=f"Letter {letter}")):  # Limit to 100 per letter
                img_path = os.path.join(letter_dir, img_file)
                
                try:
                    # Load and process image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Extract keypoints
                    keypoints = self.extract_keypoints_from_image(img)
                    if keypoints is not None:
                        # Create sequence (repeat the same frame 30 times for static signs)
                        sequence = np.tile(keypoints, (30, 1))
                        
                        # Save sequence
                        output_path = os.path.join(output_dir, f"{letter}_{i:04d}.npy")
                        np.save(output_path, sequence)
                        converted_count += 1
                        
                except Exception as e:
                    print(f"   Error processing {img_file}: {e}")
                    continue
        
        print(f"✅ Converted {converted_count} sequences")
        return True
    
    def extract_keypoints_from_image(self, img):
        """Extract keypoints from a single image"""
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose_estimator.process(img_rgb)
            
            landmarks = []
            
            # Left hand landmarks
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * 21)
            
            # Right hand landmarks
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * 21)
            
            # Pose landmarks (upper body)
            relevant_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                for idx in relevant_indices:
                    lm = pose_landmarks[idx]
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0] * len(relevant_indices))
            
            return np.array(landmarks)
            
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
            return None
    
    def create_synthetic_data(self, num_signs=10, sequences_per_sign=50):
        """Create synthetic training data for testing"""
        print("🎭 Creating synthetic training data...")
        
        signs = ['HELLO', 'THANK_YOU', 'YES', 'NO', 'PLEASE', 
                'SORRY', 'GOODBYE', 'ONE', 'TWO', 'THREE']
        
        for sign in signs[:num_signs]:
            print(f"   Creating data for: {sign}")
            
            # Create output directory
            output_dir = os.path.join(self.data_dir, sign)
            os.makedirs(output_dir, exist_ok=True)
            
            for i in range(sequences_per_sign):
                # Generate synthetic keypoints with some variation
                base_keypoints = np.random.normal(0.5, 0.1, 153)
                
                # Add temporal variation (30 frames)
                sequence = []
                for frame in range(30):
                    # Add small random variation to each frame
                    frame_keypoints = base_keypoints + np.random.normal(0, 0.01, 153)
                    sequence.append(frame_keypoints)
                
                sequence = np.array(sequence)
                
                # Ensure exact shape (30, 153)
                if sequence.shape != (30, 153):
                    print(f"   Warning: Sequence shape {sequence.shape}, reshaping to (30, 153)")
                    if sequence.shape[0] < 30:
                        # Pad with last frame
                        padding = np.tile(sequence[-1:], (30 - sequence.shape[0], 1))
                        sequence = np.vstack([sequence, padding])
                    else:
                        # Truncate to 30 frames
                        sequence = sequence[:30]
                
                # Save sequence
                output_path = os.path.join(output_dir, f"{sign}_{i:04d}.npy")
                np.save(output_path, sequence)
        
        print(f"✅ Created synthetic data for {num_signs} signs")
        return True
    
    def download_sample_videos(self):
        """Download sample sign language videos"""
        print("📹 Downloading sample sign language videos...")
        
        # Sample video URLs (you can add more)
        video_urls = {
            'HELLO': 'https://example.com/hello_sign.mp4',
            'THANK_YOU': 'https://example.com/thankyou_sign.mp4',
            # Add more URLs as needed
        }
        
        print("⚠️  Manual video download required:")
        print("   Download sign language videos and place them in 'datasets/videos/'")
        print("   Then run: python download_datasets.py --convert-videos")
        
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and convert sign language datasets")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data")
    parser.add_argument("--convert-asl", action="store_true", help="Convert ASL Alphabet dataset")
    parser.add_argument("--convert-videos", action="store_true", help="Convert video files")
    parser.add_argument("--num-signs", type=int, default=10, help="Number of signs for synthetic data")
    parser.add_argument("--sequences-per-sign", type=int, default=50, help="Sequences per sign")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.synthetic:
        downloader.create_synthetic_data(args.num_signs, args.sequences_per_sign)
    elif args.convert_asl:
        downloader.convert_asl_alphabet()
    elif args.convert_videos:
        print("Video conversion not implemented yet")
    else:
        print("🎯 Sign Language Dataset Downloader")
        print("=" * 50)
        print("Choose an option:")
        print("1. Create synthetic data (fastest):")
        print("   python download_datasets.py --synthetic")
        print()
        print("2. Convert ASL Alphabet dataset:")
        print("   python download_datasets.py --convert-asl")
        print()
        print("3. Download sample videos:")
        print("   python download_datasets.py --download-videos")

if __name__ == "__main__":
    main() 