#!/usr/bin/env python3
"""
Test Data Loading
=================

This script tests the data loading logic to ensure it works correctly.
"""

import os
import glob
import numpy as np

def test_data_loading():
    """Test the data loading logic"""
    print("🧪 Testing data loading logic...")
    print("=" * 50)
    
    # Test the exact path that will be used in Colab
    train_folders = glob.glob("data/processed/*_train")
    print(f"Found {len(train_folders)} training folders")
    
    if len(train_folders) == 0:
        print("❌ No training folders found!")
        print("   Current working directory:", os.getcwd())
        print("   Checking if data/processed exists:", os.path.exists("data/processed"))
        if os.path.exists("data/processed"):
            print("   Contents of data/processed:", os.listdir("data/processed"))
        return
    
    print(f"   Folders: {[os.path.basename(f) for f in train_folders[:5]]}...")
    
    sequences = []
    labels = []
    
    for folder in train_folders[:3]:  # Test with first 3 folders
        folder_name = os.path.basename(folder)
        letter = folder_name.split("_")[0]
        sequence_files = glob.glob(os.path.join(folder, "*.npy"))
        
        print(f"\n📁 Processing {folder_name}:")
        print(f"   Letter: {letter}")
        print(f"   Found {len(sequence_files)} .npy files")
        
        for seq_file in sequence_files[:5]:  # Test with first 5 files
            try:
                sequence = np.load(seq_file)
                if sequence.shape == (30, 153):
                    sequences.append(sequence)
                    labels.append(letter)
                    print(f"   ✅ Loaded {os.path.basename(seq_file)}: {sequence.shape}")
                else:
                    print(f"   ⚠️  Skipped {os.path.basename(seq_file)}: {sequence.shape}")
            except Exception as e:
                print(f"   ❌ Error loading {os.path.basename(seq_file)}: {e}")
                continue
    
    print(f"\n📊 Summary:")
    print(f"   Total sequences loaded: {len(sequences)}")
    print(f"   Unique labels: {sorted(set(labels))}")
    
    if len(sequences) > 0:
        print("✅ Data loading test passed!")
    else:
        print("❌ Data loading test failed!")

if __name__ == "__main__":
    test_data_loading()
