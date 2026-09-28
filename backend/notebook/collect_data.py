#!/usr/bin/env python3
"""
Sign Language Data Collection Script
===================================

This script helps you collect training data for sign language recognition.
Follow the instructions on screen to collect data for different signs.

Usage:
    python collect_data.py
"""

import sys
import os

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_collector import main

if __name__ == "__main__":
    print("🎯 Sign Language Data Collection")
    print("=" * 50)
    print("This will help you collect training data for sign language recognition.")
    print("Make sure you have a webcam connected and good lighting.")
    print("=" * 50)
    
    input("Press Enter to start data collection...")
    
    main() 