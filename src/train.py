# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

from config import Config
from model import IndoorSegmentationModel, IndoorSegmentationLoss
from data_prep import get_dataloaders

class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = IndoorSegmentationModel().to(self.device)
        
        # Loss function
        self.criterion = IndoorSegmentationLoss(use_dice_loss=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            # Forward pass
            outputs = self.model(images)
            loss, ce_loss, dice_loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}',
                'dice': f'{dice_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce_loss / len(train_loader)
        avg_dice = total_dice_loss / len(train_loader)
        
        return avg_loss, avg_ce, avg_dice
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_dice_loss = 0
        all_ious = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                # Forward pass
                outputs = self.model(images)
                loss, ce_loss, dice_loss = self.criterion(outputs, masks)
                
                # Calculate IoU
                preds = torch.argmax(outputs, dim=1)
                iou = self.calculate_iou(preds, masks)
                all_ious.append(iou)
                
                # Update metrics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_dice_loss += dice_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_ce = total_ce_loss / len(val_loader)
        avg_dice = total_dice_loss / len(val_loader)
        avg_iou = np.mean(all_ious)
        
        return avg_loss, avg_ce, avg_dice, avg_iou
    
    def calculate_iou(self, preds, targets, num_classes=4):
        """Calculate mean IoU"""
        ious = []
        
        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()
            
            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(float('nan'))
        
        return np.nanmean(ious)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(Config.MODEL_SAVE_PATH, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_ious = checkpoint['val_ious']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0].plot(self.val_losses, label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IoU plot
        axes[1].plot(self.val_ious, label='Val IoU', color='green', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('Validation IoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(Config.OUTPUT_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history saved to '{plot_path}'")
        plt.show()
    
    def train(self):
        """Main training loop"""
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
            num_workers=Config.NUM_WORKERS
        )
        
        start_time = time.time()
        
        for epoch in range(1, Config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{Config.EPOCHS}")
            print("-" * 40)
            
            # Train
            train_loss, train_ce, train_dice = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_ce, val_dice, val_iou = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Dice: {train_dice:.4f})")
            print(f"Val Loss: {val_loss:.4f} (CE: {val_ce:.4f}, Dice: {val_dice:.4f})")
            print(f"Val IoU: {val_iou:.4f}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % Config.SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation IoU: {max(self.val_ious):.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.model

def main():
    """Main function"""
    trainer = Trainer()
    model = trainer.train()
    
    print("\n✓ Training script completed!")
    print(f"Model saved in: {Config.MODEL_SAVE_PATH}")
    print(f"Outputs saved in: {Config.OUTPUT_PATH}")

if __name__ == "__main__":
    main()
