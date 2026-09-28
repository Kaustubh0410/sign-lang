import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from translator import GestureTransformer
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pickle
import random

class SignLanguageDataset(Dataset):
    """Dataset for sign language sequences with data augmentation"""
    
    def __init__(self, sequences, labels, augment=True):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Data augmentation
        if self.augment and random.random() < 0.5:
            sequence = self._augment_sequence(sequence)
            
        return torch.FloatTensor(sequence), self.labels[idx]
    
    def _augment_sequence(self, sequence):
        """Apply data augmentation to sequence"""
        # Add small random noise
        noise = np.random.normal(0, 0.01, sequence.shape)
        sequence = sequence + noise
        
        # Random time warping (slight speed variation) - but maintain exact length
        if random.random() < 0.3:
            # Simple interpolation to maintain exact 30 frames
            indices = np.linspace(0, sequence.shape[0]-1, sequence.shape[0])
            # Add small random variation to indices
            indices += np.random.normal(0, 0.1, indices.shape)
            indices = np.clip(indices, 0, sequence.shape[0]-1)
            
            # Interpolate to get new sequence
            new_sequence = []
            for i in range(sequence.shape[0]):
                idx = int(indices[i])
                new_sequence.append(sequence[idx])
            sequence = np.array(new_sequence)
        
        # Ensure exact shape (30, 153)
        if sequence.shape != (30, 153):
            if sequence.shape[0] < 30:
                # Pad with last frame
                padding = np.tile(sequence[-1:], (30 - sequence.shape[0], 1))
                sequence = np.vstack([sequence, padding])
            else:
                # Truncate to 30 frames
                sequence = sequence[:30]
        
        return sequence

class SignLanguageTrainer:
    """Train the sign language recognition model with improved efficiency"""
    
    def __init__(self, data_dir="data/raw", model_dir="models", sequence_length=30):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Using device: {self.device}")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Optimized training parameters for efficiency
        self.batch_size = 64  # Increased for better GPU utilization
        self.learning_rate = 0.0005  # Slightly lower for stability
        self.epochs = 150  # More epochs with early stopping
        self.patience = 15  # Increased patience
        
        # Model architecture parameters
        self.hidden_dim = 256
        self.num_heads = 8
        self.num_layers = 4
        
    def load_data(self):
        """Load training data from collected sequences with validation"""
        print("📂 Loading training data...")
        
        sequences = []
        labels = []
        
        # Walk through data directory
        for sign_name in os.listdir(self.data_dir):
            sign_dir = os.path.join(self.data_dir, sign_name)
            if not os.path.isdir(sign_dir):
                continue
                
            print(f"   Loading {sign_name}...")
            
            # Load all sequences for this sign
            for filename in os.listdir(sign_dir):
                if filename.endswith('.npy'):
                    filepath = os.path.join(sign_dir, filename)
                    try:
                        sequence = np.load(filepath)
                        
                        # Validate sequence shape
                        if sequence.shape == (self.sequence_length, 153):
                            sequences.append(sequence)
                            labels.append(sign_name)
                        else:
                            print(f"   ⚠️  Skipping {filename}: wrong shape {sequence.shape}, expected ({self.sequence_length}, 153)")
                    except Exception as e:
                        print(f"   ❌ Error loading {filename}: {e}")
        
        if len(sequences) == 0:
            raise ValueError("No valid training data found! Please collect data first.")
        
        print(f"✅ Loaded {len(sequences)} sequences for {len(set(labels))} signs")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Save label mapping
        label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        with open(os.path.join(self.model_dir, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"📊 Label mapping: {label_mapping}")
        
        return np.array(sequences), encoded_labels
    
    def prepare_data(self, sequences, labels):
        """Prepare data for training"""
        print("🔧 Preparing data for training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = SignLanguageDataset(X_train, y_train, augment=True)
        test_dataset = SignLanguageDataset(X_test, y_test, augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        return train_loader, test_loader
    
    def create_model(self, num_classes):
        """Create the model"""
        print(f"🏗️  Creating model with {num_classes} classes...")
        
        model = GestureTransformer(
            input_dim=153,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        return model
    
    def train_model(self, train_loader, test_loader, num_classes):
        """Train the model with improved efficiency"""
        print("🚀 Starting model training...")
        
        # Create model
        model = self.create_model(num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, verbose=True)
        
        # Training history
        train_losses = []
        test_losses = []
        test_accuracies = []
        best_accuracy = 0
        patience_counter = 0
        
        print(f"📊 Training for {self.epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (batch_sequences, batch_labels) in enumerate(train_loader):
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Evaluation phase
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_sequences, batch_labels in test_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_sequences)
                    loss = criterion(outputs, batch_labels)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            train_accuracy = 100 * train_correct / train_total
            test_accuracy = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}]")
                print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
                print(f"   Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
                print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                print("-" * 40)
            
            # Save best model
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'label_encoder': self.label_encoder
                }, os.path.join(self.model_dir, 'best_gesture_model.pt'))
                print(f"💾 Saved best model with accuracy: {test_accuracy:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'final_accuracy': test_accuracy,
            'label_encoder': self.label_encoder
        }, os.path.join(self.model_dir, 'final_gesture_model.pt'))
        
        # Plot training history
        self.plot_training_history(train_losses, test_losses, test_accuracies)
        
        print(f"✅ Training completed!")
        print(f"   Best accuracy: {best_accuracy:.2f}%")
        print(f"   Final accuracy: {test_accuracy:.2f}%")
        
        return model
    
    def plot_training_history(self, train_losses, test_losses, test_accuracies):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        plt.close()
        
        print(f"📈 Training history saved to: {os.path.join(self.model_dir, 'training_history.png')}")
    
    def evaluate_model(self, model, test_loader):
        """Evaluate the trained model"""
        print("🔍 Evaluating model...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                outputs = model(batch_sequences)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate accuracy
        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = 100 * correct / len(all_labels)
        
        print(f"📊 Model Evaluation:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Correct: {correct}/{len(all_labels)}")
        
        return accuracy
    
    def train(self):
        """Main training function"""
        print("🎯 Sign Language Model Training")
        print("=" * 50)
        
        # Load data
        sequences, labels = self.load_data()
        
        if len(sequences) == 0:
            print("❌ No training data found!")
            print("   Please collect data first using the data collector.")
            return
        
        # Prepare data
        train_loader, test_loader = self.prepare_data(sequences, labels)
        
        # Train model
        num_classes = len(self.label_encoder.classes_)
        model = self.train_model(train_loader, test_loader, num_classes)
        
        # Evaluate model
        accuracy = self.evaluate_model(model, test_loader)
        
        print(f"🎉 Training completed with {accuracy:.2f}% accuracy!")

def main():
    """Main function to run training"""
    trainer = SignLanguageTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 