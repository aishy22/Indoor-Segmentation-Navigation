# src/train.py 
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

from config import Config
from model import IndoorSegmentationModel
from data_prep import get_dataloaders

class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Create model with MORE dropout for regularization
        self.model = IndoorSegmentationModel().to(self.device)
        
        # Add more regularization: Dropout in the model itself
        # (We'll modify model.py separately)
        
        # Class weights to handle imbalance
        class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # L2 Regularization (weight_decay) - INCREASED
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=1e-3  # Increased from 1e-4 (stronger regularization)
        )
        
        # Reduce LR more aggressively when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, cooldown=1
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stop_patience = 5  # Stop if no improvement for 5 epochs
        
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    def train_epoch(self, train_loader):
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(val_loader)
    
    def train(self):
        print("=" * 60)
        print("STARTING TRAINING WITH REGULARIZATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Batch size: {Config.BATCH_SIZE}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print(f"Weight decay: 1e-3 (L2 regularization)")
        print(f"Early stopping patience: {self.early_stop_patience}")
        print("=" * 60)
        
        # Get dataloaders
        print("\n📁 Loading datasets...")
        train_loader, val_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            max_train_samples=None,
            max_val_samples=None
        )
        
        print(f"\n📊 Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
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
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Print results
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Learning Rate: {new_lr:.6f}")
            
            if new_lr < old_lr:
                print(f"   📉 Learning rate reduced!")
            
            # Calculate gap
            gap = val_loss - train_loss
            print(f"   Gap (Val - Train): {gap:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"   ✅ Saved best model! (val_loss: {self.best_val_loss:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"   ⚠️ No improvement for {self.patience_counter} epochs")
            
            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {self.early_stop_patience} epochs.")
                break
            
            # Warning for overfitting
            if gap > 0.2:
                print(f"   ⚠️ Warning: Large gap ({gap:.4f}) - Possible overfitting!")
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')}")
        
        self.plot_training_history()
        return self.model
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight overfitting region
        if len(self.val_losses) > 3:
            val_arr = np.array(self.val_losses)
            if len(val_arr) > 3 and val_arr[-1] > val_arr[-2]:
                plt.axvspan(len(self.val_losses)-2, len(self.val_losses), 
                           alpha=0.3, color='red', label='Overfitting region')
        
        plt.subplot(1, 2, 2)
        gaps = [self.val_losses[i] - self.train_losses[i] for i in range(len(self.train_losses))]
        plt.plot(gaps, label='Val - Train Gap', color='orange', linewidth=2)
        plt.axhline(y=0.1, color='r', linestyle='--', label='Warning threshold')
        plt.xlabel('Epoch')
        plt.ylabel('Gap')
        plt.title('Overfitting Monitor')
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
