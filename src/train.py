# src/train.py

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

try:
    from config import Config
except ImportError:
    class Config:
        BASE_PATH = '/content/Indoor-Segmentation-Navigation'
        MODEL_SAVE_PATH = '/content/Indoor-Segmentation-Navigation/models'
        OUTPUT_PATH = '/content/Indoor-Segmentation-Navigation/outputs'
        BATCH_SIZE = 8
        EPOCHS = 15
        LEARNING_RATE = 1e-4
        NUM_WORKERS = 2
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        SAVE_INTERVAL = 5

from model import IndoorSegmentationModel, IndoorSegmentationLoss
from data_prep import get_dataloaders

# Create directories
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = IndoorSegmentationModel().to(self.device)
        
        # Loss function with class weights
        class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(self.device)
        self.criterion = IndoorSegmentationLoss(use_dice_loss=True, class_weights=class_weights)
        
        # Optimizer
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
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, ce_loss, dice_loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                outputs = self.model(images)
                loss, ce_loss, dice_loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
    
    def train(self):
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        # Get dataloaders
        train_loader, val_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            max_train_samples=500,  # Use 500 training images for quick testing
            max_val_samples=100      # Use 100 validation images
        )
        
        start_time = time.time()
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{Config.EPOCHS}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth'))
                print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            # Save checkpoint every SAVE_INTERVAL epochs
            if epoch % Config.SAVE_INTERVAL == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_val_loss': self.best_val_loss
                }
                torch.save(checkpoint, os.path.join(Config.MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch}.pth'))
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.model
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.val_losses, label='Val Loss', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_PATH, 'training_history.png'))
        plt.show()


def main():
    trainer = Trainer()
    model = trainer.train()
    print("\n✓ Training script completed!")
    print(f"Model saved in: {Config.MODEL_SAVE_PATH}")
    print(f"Outputs saved in: {Config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
