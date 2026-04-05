# src/train.py - NEW VERSION
# Trains on full ADE20K dataset (20,210 training images, 2,000 validation images)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Add paths
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

from config import Config
from model import IndoorSegmentationModel
from data_prep import get_dataloaders

class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Create model with pre-trained weights
        self.model = IndoorSegmentationModel().to(self.device)
        
        # Class weights to handle imbalance (floor, wall, door, no-go)
        class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create directories
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("STARTING TRAINING ON FULL DATASET")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        # Get dataloaders (using ALL images - no max_samples limit)
        print("\n📁 Loading datasets...")
        train_loader, val_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            max_train_samples=None,  # Use ALL training images
            max_val_samples=None      # Use ALL validation images
        )
        
        print(f"\n📊 Training batches: {len(train_loader)} ({len(train_loader) * Config.BATCH_SIZE:,} images)")
        print(f"   Validation batches: {len(val_loader)} ({len(val_loader) * Config.BATCH_SIZE:,} images)")
        
        start_time = time.time()
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{Config.EPOCHS}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Print results
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"   ✅ Saved best model! (val_loss: {self.best_val_loss:.4f})")
            
            # Early stopping (optional)
            if epoch > 10 and val_loss > min(self.val_losses[-5:]):
                print(f"\n⚠️ Validation loss stopped improving. Consider stopping early.")
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.model
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Val Loss', color='green', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(Config.OUTPUT_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n📈 Training history saved to: {plot_path}")
        plt.show()


def main():
    trainer = Trainer()
    model = trainer.train()
    print("\n✅ Training script completed!")


if __name__ == "__main__":
    main()
