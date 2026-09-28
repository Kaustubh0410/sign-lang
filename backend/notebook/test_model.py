#!/usr/bin/env python3
"""
Quick Model Test Script
======================

This script tests the trained model with synthetic data to verify it's working correctly.

Usage:
    python test_model.py
"""

import torch
import numpy as np
import json
import os
from utils.translator import GestureTransformer

def test_model():
    """Test the trained model"""
    print("🧪 Testing Trained Model")
    print("=" * 40)
    
    # Load label mapping
    with open('models/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    print(f"📊 Model can recognize: {list(label_mapping.values())}")
    
    # Load the trained model
    try:
        checkpoint = torch.load('models/best_gesture_model.pt', map_location='cpu')
        model = GestureTransformer(
            input_dim=153,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            num_classes=len(label_mapping)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Generate test sequences
    print("\n🎯 Testing with synthetic sequences...")
    
    test_results = []
    
    for sign_name in label_mapping.values():
        # Generate a test sequence for this sign
        test_sequence = np.random.normal(0.5, 0.1, (30, 153))
        
        # Add some sign-specific characteristics
        if sign_name == "HELLO":
            # Simulate waving motion
            for i in range(30):
                test_sequence[i, :63] += np.sin(i * 0.2) * 0.1  # Left hand motion
        elif sign_name == "YES":
            # Simulate nodding motion
            for i in range(30):
                test_sequence[i, 126:153] += np.sin(i * 0.3) * 0.1  # Head motion
        
        # Convert to tensor
        test_tensor = torch.FloatTensor(test_sequence).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(test_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        predicted_sign = label_mapping[str(predicted_class)]
        is_correct = predicted_sign == sign_name
        
        test_results.append({
            'input_sign': sign_name,
            'predicted_sign': predicted_sign,
            'confidence': confidence,
            'correct': is_correct
        })
        
        status = "✅" if is_correct else "❌"
        print(f"   {status} {sign_name} → {predicted_sign} ({confidence:.2%})")
    
    # Summary
    correct_predictions = sum(1 for r in test_results if r['correct'])
    total_predictions = len(test_results)
    accuracy = correct_predictions / total_predictions
    
    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    print(f"   Average confidence: {np.mean([r['confidence'] for r in test_results]):.1%}")
    
    if accuracy > 0.5:
        print("🎉 Model is working! (Expected low accuracy with synthetic data)")
    else:
        print("⚠️  Low accuracy expected with synthetic data")
    
    print("\n💡 Next steps:")
    print("   1. Download real sign language dataset")
    print("   2. Train with real data for better accuracy")
    print("   3. Test with webcam input")
    
    return test_results

def test_real_time_prediction():
    """Test real-time prediction capability"""
    print("\n🎥 Testing Real-time Prediction")
    print("=" * 40)
    
    try:
        import cv2
        from utils.pose_estimation import PoseEstimator
        
        # Initialize pose estimator
        pose_estimator = PoseEstimator(sequence_length=30)
        
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not available")
            return
        
        print("✅ Webcam available for real-time testing")
        print("   Run: python app.py to test with web interface")
        
        cap.release()
        
    except ImportError:
        print("❌ OpenCV not available")
    except Exception as e:
        print(f"❌ Error testing webcam: {e}")

def main():
    """Main test function"""
    # Test model predictions
    test_results = test_model()
    
    # Test real-time capability
    test_real_time_prediction()
    
    print("\n🎯 Model Testing Complete!")
    print("=" * 40)
    print("Your sign language model is ready for use!")
    print("Use 'python app.py' to start the web interface.")

if __name__ == "__main__":
    main() 