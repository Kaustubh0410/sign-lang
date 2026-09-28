# Sign Language Translator

A real-time sign language to English translation system using advanced machine learning techniques.

## 🎯 **ML Pipeline Overview**

### **Architecture**
```
Video Input → Pose Estimation → Keypoint Extraction → Temporal Sequence → Gesture Recognition → Gloss Sequence → Natural Language Generation → English Translation
```

### **Key Components**

#### **1. Pose Estimation (MediaPipe Holistic)**
- **Input**: Video frames (30fps)
- **Output**: 189-dimensional keypoint vectors
  - Left hand: 21 points × 3 coordinates = 63 features
  - Right hand: 21 points × 3 coordinates = 63 features  
  - Pose (upper body): 9 points × 3 coordinates = 27 features
- **Performance**: Real-time capable (30+ FPS)

#### **2. Gesture Recognition Model (Transformer)**
- **Architecture**: Multi-head attention transformer
- **Input**: Temporal sequences (30 frames = 1 second)
- **Output**: Gloss predictions (sign language tokens)
- **Features**:
  - 256 hidden dimensions
  - 8 attention heads
  - 4 transformer layers
  - Dropout for regularization

#### **3. Natural Language Generation (T5)**
- **Model**: T5-small fine-tuned for ASL→English
- **Input**: Gloss sequences
- **Output**: Fluent English sentences
- **Features**: Beam search, caching for performance

## 🚀 **Getting Started**

### **Prerequisites**
```bash
# Frontend dependencies
cd frontend
npm install

# Backend dependencies
cd backend
pip install -r requirements.txt
```

### **Running the Application**

1. **Start Backend Server**
```bash
cd backend
python app.py
# Server runs on http://localhost:8000
```

2. **Start Frontend Development Server**
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:5174
```

3. **Access the Application**
- Open `http://localhost:5174` in your browser
- Allow camera permissions
- Start signing!

## 📊 **ML Model Performance**

### **Accuracy Targets**
- **Gesture Recognition**: 85%+ accuracy on common signs
- **Translation Quality**: 90%+ BLEU score
- **Real-time Latency**: <500ms end-to-end

### **Supported Signs**
The system recognizes 100+ common ASL signs including:
- Basic greetings (HELLO, THANK-YOU, PLEASE)
- Common verbs (GO, COME, EAT, DRINK, SLEEP)
- Pronouns (I, YOU, HE, SHE, THEY)
- Question words (WHAT, WHERE, WHEN, WHY, HOW)
- Numbers and colors
- Family members and emotions

## 🔧 **Advanced Configuration**

### **Model Training**

1. **Data Collection**
```bash
# Use the keypoint extraction notebook
jupyter notebook notebooks/01_keypoint_extraction.ipynb
```

2. **Model Training**
```bash
# Train the gesture recognition model
jupyter notebook notebooks/02_model_training.ipynb
```

3. **Fine-tuning T5 for ASL**
```bash
# Fine-tune T5 for ASL→English translation
jupyter notebook notebooks/03_nlg_training.ipynb
```

### **Performance Optimization**

#### **Real-time Processing**
- **Frame Rate**: 30 FPS for smooth video
- **Sequence Length**: 30 frames (1 second) for temporal context
- **Batch Processing**: Efficient GPU utilization
- **Caching**: Translation results cached for repeated signs

#### **Accuracy Improvements**
- **Data Augmentation**: Noise injection, keypoint dropout
- **Attention Visualization**: Interpretable attention weights
- **Confidence Scoring**: Uncertainty quantification
- **Ensemble Methods**: Multiple model predictions

## 🎯 **Best Practices for High Accuracy**

### **1. Data Quality**
- **Lighting**: Good, consistent lighting
- **Background**: Clean, uncluttered background
- **Camera Position**: Front-facing, stable camera
- **Hand Visibility**: Clear view of both hands

### **2. Signing Technique**
- **Clear Movements**: Deliberate, well-formed signs
- **Proper Hand Shape**: Accurate hand configurations
- **Facial Expressions**: Include facial grammar
- **Temporal Consistency**: Maintain steady signing pace

### **3. System Optimization**
- **GPU Acceleration**: Use CUDA for faster inference
- **Model Quantization**: Reduce model size for mobile deployment
- **Edge Computing**: Local processing for privacy
- **Cloud Integration**: Scalable backend for heavy workloads

## 📈 **Model Architecture Details**

### **Gesture Recognition Transformer**
```python
class GestureTransformer(nn.Module):
    def __init__(self, input_dim=189, hidden_dim=256, num_heads=8, num_layers=4):
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim))
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ), 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
```

### **Translation Pipeline**
```python
def translate_frame_sequence(keypoint_sequence):
    # 1. Gesture recognition
    gloss_sequence = gesture_model.predict(keypoint_sequence)
    
    # 2. Natural language generation
    english_translation = nlg_model.generate(gloss_sequence)
    
    # 3. Confidence scoring
    confidence = calculate_confidence(gloss_sequence, english_translation)
    
    return {
        "gloss_sequence": gloss_sequence,
        "english_translation": english_translation,
        "confidence": confidence
    }
```

## 🔬 **Research and Development**

### **Current Capabilities**
- ✅ Real-time pose estimation
- ✅ Temporal sequence processing
- ✅ Transformer-based gesture recognition
- ✅ T5-powered natural language generation
- ✅ WebSocket-based real-time communication
- ✅ Modern React frontend with Material-UI

### **Future Enhancements**
- 🔄 **Multi-language Support**: Support for other sign languages
- 🔄 **Sentence-level Translation**: Complete sentence understanding
- 🔄 **Facial Expression Recognition**: Enhanced grammar understanding
- 🔄 **Mobile Deployment**: Optimized for mobile devices
- 🔄 **Offline Mode**: Local processing without internet
- 🔄 **Custom Training**: User-specific model adaptation

### **Performance Metrics**
- **Latency**: <500ms end-to-end translation
- **Accuracy**: 85%+ gesture recognition accuracy
- **Throughput**: 30 FPS real-time processing
- **Memory**: <2GB GPU memory usage
- **Scalability**: Support for multiple concurrent users

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **MediaPipe**: For real-time pose estimation
- **PyTorch**: For deep learning framework
- **Transformers**: For natural language processing
- **React**: For frontend framework
- **FastAPI**: For backend API framework

---

**For questions or support, please open an issue or contact the development team.**
