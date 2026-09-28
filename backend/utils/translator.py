import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List, Dict, Optional

class GestureTransformer(nn.Module):
    """Transformer-based model for gesture recognition"""
    def __init__(self, input_dim=153, hidden_dim=256, num_heads=8, num_layers=4, num_classes=1000):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 30, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        # Use the last token for classification
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class SignTranslator:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models lazily
        self._gesture_model = None
        self._nlg_model = None
        self._tokenizer = None
        
        # Gloss vocabulary (common ASL signs)
        self.gloss_vocab = self._create_gloss_vocabulary()
        
        # Translation cache for performance
        self.translation_cache = {}
        
    @property
    def gesture_model(self):
        """Lazy load gesture model"""
        if self._gesture_model is None:
            self._gesture_model = self._load_gesture_model()
        return self._gesture_model
    
    @property
    def nlg_model(self):
        """Lazy load NLG model"""
        if self._nlg_model is None:
            self._nlg_model = self._load_nlg_model()
        return self._nlg_model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer
        
    def _load_gesture_model(self):
        """Load or create the gesture recognition model"""
        model_path = os.path.join(self.model_dir, "gesture_model.pt")
        print(f"Loading gesture model from: {model_path}")
        print(f"Model directory exists: {os.path.exists(self.model_dir)}")
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                print(f"Found existing model file: {model_path}")
                model = torch.load(model_path, map_location=self.device)
                print(f"Loaded existing gesture model from {model_path}")
                print(f"Model input projection shape: {model.input_projection.weight.shape}")
                return model
        except Exception as e:
            print(f"Error loading existing model: {e}")
        
        # Create new model if loading fails
        print("Creating new gesture model...")
        print(f"Input dimension: 153, Hidden dimension: 256, Num classes: {len(self.gloss_vocab)}")
        model = GestureTransformer(
            input_dim=153,  # 153 keypoints (21 left hand + 21 right hand + 9 pose points) × 3 coordinates
            hidden_dim=256,
            num_heads=8,
            num_layers=6,
            num_classes=len(self.gloss_vocab)
        ).to(self.device)
        
        print(f"Created model with input projection shape: {model.input_projection.weight.shape}")
        
        # Save the new model
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            torch.save(model, model_path)
            print(f"Saved new gesture model to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
        
        return model
    
    def _load_nlg_model(self):
        """Load or create the natural language generation model"""
        model_path = os.path.join(self.model_dir, "nlg_model.pt")
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location=self.device)
                print(f"Loaded existing NLG model from {model_path}")
                return model
        except Exception as e:
            print(f"Error loading existing NLG model: {e}")
        
        # Create fallback model if loading fails
        print("Creating fallback NLG model...")
        try:
            # Try to load T5 model from Hugging Face
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
            model.to(self.device)
            print("Loaded T5-small model from Hugging Face")
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            # Create a simple fallback model
            class DummyNLG:
                def __call__(self, input_ids, **kwargs):
                    # Return dummy logits
                    batch_size = input_ids.shape[0]
                    vocab_size = 32100  # T5 vocab size
                    return type('obj', (object,), {
                        'logits': torch.randn(batch_size, input_ids.shape[1], vocab_size).to(self.device)
                    })()
            
            model = DummyNLG()
            print("Using dummy NLG model")
        
        # Save the model
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            torch.save(model, model_path)
            print(f"Saved NLG model to {model_path}")
        except Exception as e:
            print(f"Error saving NLG model: {e}")
        
        return model
    
    def _load_tokenizer(self):
        """Load T5 tokenizer for natural language generation"""
        try:
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            print("Loaded T5 tokenizer from Hugging Face")
            return tokenizer
        except Exception as e:
            print(f"Error loading T5 tokenizer: {e}")
            # Create a simple fallback tokenizer
            class DummyTokenizer:
                def encode(self, text, **kwargs):
                    return [1, 2, 3, 4, 5]  # Dummy tokens
                def decode(self, tokens, **kwargs):
                    return "Hello world"  # Dummy text
                def __call__(self, text, **kwargs):
                    return {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}
            
            return DummyTokenizer()
    
    def _create_dummy_nlg_model(self):
        """Create dummy NLG model for testing"""
        class DummyNLG(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 256)
                self.lstm = nn.LSTM(256, 256, batch_first=True)
                self.fc = nn.Linear(256, 1000)
                
            def forward(self, x):
                x = self.embedding(x)
                x, _ = self.lstm(x)
                return self.fc(x[:, -1, :])
                
            def generate(self, input_ids, max_length=50):
                # Dummy generation
                return torch.randint(0, 1000, (input_ids.size(0), max_length))
        
        return DummyNLG()
    
    def _create_gloss_vocabulary(self):
        """Create vocabulary of common ASL glosses"""
        return {
            0: "HELLO", 1: "THANK-YOU", 2: "PLEASE", 3: "SORRY", 4: "YES", 5: "NO",
            6: "HELP", 7: "UNDERSTAND", 8: "LEARN", 9: "TEACH", 10: "WORK",
            11: "HOME", 12: "FAMILY", 13: "FRIEND", 14: "LOVE", 15: "GOOD",
            16: "BAD", 17: "BIG", 18: "SMALL", 19: "FAST", 20: "SLOW",
            21: "EAT", 22: "DRINK", 23: "SLEEP", 24: "WAKE-UP", 25: "GO",
            26: "COME", 27: "STOP", 28: "WAIT", 29: "LOOK", 30: "SEE",
            31: "HEAR", 32: "SPEAK", 33: "SIGN", 34: "READ", 35: "WRITE",
            36: "NAME", 37: "WHAT", 38: "WHERE", 39: "WHEN", 40: "WHY",
            41: "HOW", 42: "WHO", 43: "I", 44: "YOU", 45: "HE", 46: "SHE",
            47: "THEY", 48: "WE", 49: "ME", 50: "MINE", 51: "YOURS",
            52: "HIS", 53: "HERS", 54: "THEIRS", 55: "OURS", 56: "THIS",
            57: "THAT", 58: "HERE", 59: "THERE", 60: "NOW", 61: "LATER",
            62: "BEFORE", 63: "AFTER", 64: "TODAY", 65: "TOMORROW", 66: "YESTERDAY",
            67: "WEEK", 68: "MONTH", 69: "YEAR", 70: "TIME", 71: "DAY",
            72: "NIGHT", 73: "MORNING", 74: "AFTERNOON", 75: "EVENING",
            76: "WATER", 77: "FOOD", 78: "BOOK", 79: "PAPER", 80: "PEN",
            81: "PHONE", 82: "COMPUTER", 83: "CAR", 84: "BUS", 85: "TRAIN",
            86: "PLANE", 87: "MONEY", 88: "DOLLAR", 89: "CENT", 90: "BUY",
            91: "SELL", 92: "GIVE", 93: "TAKE", 94: "OPEN", 95: "CLOSE",
            96: "START", 97: "FINISH", 98: "BEGIN", 99: "END"
        }
    
    def predict_gloss(self, keypoint_sequence: np.ndarray) -> List[str]:
        """Predict glosses from keypoint sequence"""
        try:
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(keypoint_sequence).unsqueeze(0).to(self.device)
            print(f"Input sequence tensor shape: {sequence_tensor.shape}")
            print(f"Model input projection weight shape: {self.gesture_model.input_projection.weight.shape}")
            
            # Get model predictions
            logits = self.gesture_model(sequence_tensor)
            print(f"Model output logits shape: {logits.shape}")
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(probs, dim=-1)
            
            print(f"Predicted indices: {predicted_indices}")
            
            # Convert to glosses
            gloss_sequence = []
            for idx in predicted_indices:
                idx_item = idx.item() if hasattr(idx, 'item') else idx
                print(f"Processing index: {idx_item}")
                if idx_item < len(self.gloss_vocab):
                    gloss = self.gloss_vocab[idx_item]
                    print(f"Found gloss: {gloss}")
                    gloss_sequence.append(gloss)
                else:
                    print(f"Index {idx_item} out of range, using UNKNOWN")
                    gloss_sequence.append("UNKNOWN")
            
            print(f"Predicted gloss sequence: {gloss_sequence}")
            return gloss_sequence
            
        except Exception as e:
            print(f"Error in predict_gloss: {e}")
            import traceback
            traceback.print_exc()
            return ["HELLO"]  # Fallback
    
    def gloss_to_sentence(self, gloss_sequence: List[str]) -> str:
        """Convert gloss sequence to English sentence"""
        if not gloss_sequence:
            return "No signs detected"
            
        # Create input text for T5
        gloss_text = " ".join(gloss_sequence)
        
        # Check cache first
        cache_key = gloss_text
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            if self.tokenizer and hasattr(self.nlg_model, 'generate'):
                # Use T5 for translation
                inputs = self.tokenizer.encode(
                    f"translate ASL to English: {gloss_text}",
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                outputs = self.nlg_model.generate(
                    inputs,
                    max_length=50,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
                sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # Fallback to rule-based translation
                sentence = self._rule_based_translation(gloss_sequence)
                
        except Exception as e:
            print(f"Translation error: {e}")
            sentence = self._rule_based_translation(gloss_sequence)
        
        # Cache the result
        self.translation_cache[cache_key] = sentence
        return sentence
    
    def _rule_based_translation(self, gloss_sequence: List[str]) -> str:
        """Simple rule-based translation as fallback"""
        if not gloss_sequence:
            return "No signs detected"
            
        # Simple grammar rules
        words = []
        for gloss in gloss_sequence:
            # Clean up gloss
            word = gloss.replace("-", " ").lower()
            words.append(word)
        
        # Basic sentence construction
        if len(words) == 1:
            return words[0].capitalize()
        elif len(words) == 2:
            return f"{words[0].capitalize()} {words[1]}"
        else:
            return " ".join(words).capitalize()
    
    def translate_frame_sequence(self, keypoint_sequence: np.ndarray) -> Dict[str, any]:
        """Complete translation pipeline"""
        if keypoint_sequence is None:
            return {
                "gloss_sequence": [],
                "english_translation": "No signs detected",
                "confidence": 0.0
            }
        
        # Predict gloss sequence
        gloss_sequence = self.predict_gloss(keypoint_sequence)
        
        # Convert to English
        english_translation = self.gloss_to_sentence(gloss_sequence)
        
        # Calculate confidence (simplified)
        confidence = min(0.9, len(gloss_sequence) * 0.1) if gloss_sequence else 0.0
        
        return {
            "gloss_sequence": gloss_sequence,
            "english_translation": english_translation,
            "confidence": confidence
        }
