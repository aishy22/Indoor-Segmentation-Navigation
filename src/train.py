# src/train.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Path setup for Colab
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

from config import Config
from data_prep import get_dataloaders
from model import create_model, create_loss


# ============================================
# mIoU METRIC
# ============================================
def compute_miou(preds, targets, num_classes=4):
    """
    Compute mean Intersection over Union across all classes.
    Args:
        preds  : (B, H, W) predicted class indices
        targets: (B, H, W) ground truth class indices
    Returns:
        miou: float — mean IoU across all classes
    """
    ious = []
    preds   = preds.cpu().numpy().flatten()
    targets = targets.cpu().numpy().flatten()

    for cls in range(num_classes):
        pred_cls   = (preds == cls)
        target_cls = (targets == cls)

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union        = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            # Class not present in this batch — skip it
            continue
        ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


# ============================================
# MAIN TRAINING
# ============================================
def main():
    # Setup directories and print config
    Config.setup()
    Config.print_config()

    # ── Data ──────────────────────────────────
    print("\n📁 Creating datasets...")
    train_loader, val_loader = get_dataloaders()

    # ── Model ─────────────────────────────────
    print(f"\n🔧 Using device: {Config.DEVICE}")
    model = create_model(pretrained=True).to(Config.DEVICE)

    # ── Loss ──────────────────────────────────
    # Upweight rare classes: door (1.5) and no-go (2.0)
    class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(Config.DEVICE)
    criterion = create_loss(class_weights=class_weights)

    # ── Optimizer & Scheduler ─────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ── Training Loop ─────────────────────────
    best_val_loss = float('inf')
    train_losses, val_losses, val_mious = [], [], []

    print("\n" + "=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"   Training batches  : {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Epochs            : {Config.EPOCHS}")
    print("=" * 60)

    for epoch in range(1, Config.EPOCHS + 1):

        # ── Train ───────────────────────────────
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Train]")
        for images, masks in pbar:
            images = images.to(Config.DEVICE)
            masks  = masks.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs             = model(images)
            total, ce, dice     = criterion(outputs, masks)
            total.backward()
            optimizer.step()

            train_loss += total.item()
            pbar.set_postfix({
                'loss': f'{total.item():.4f}',
                'ce'  : f'{ce.item():.4f}',
                'dice': f'{dice.item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validate ────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{Config.EPOCHS} [Val]"):
                images = images.to(Config.DEVICE)
                masks  = masks.to(Config.DEVICE)

                outputs          = model(images)
                total, ce, dice  = criterion(outputs, masks)
                val_loss        += total.item()

                # Collect predictions for mIoU
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_targets.append(masks)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Compute mIoU over full validation set
        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        miou        = compute_miou(all_preds, all_targets, num_classes=Config.NUM_CLASSES)
        val_mious.append(miou)

        scheduler.step(avg_val_loss)

        print(
            f"\nEpoch {epoch:02d}/{Config.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mIoU: {miou:.4f}"
        )

        # ── Save best checkpoint ─────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss'       : best_val_loss,
                'val_miou'            : miou,
            }, checkpoint_path)
            print(f"  ✓ Saved best model → val_loss: {best_val_loss:.4f} | mIoU: {miou:.4f}")

    print(f"\n✅ Training complete!")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Best val mIoU : {max(val_mious):.4f}")

    # ── Plot Training History ──────────────────
    epochs_range = range(1, Config.EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, train_losses, label='Train Loss', marker='o')
    axes[0].plot(epochs_range, val_losses,   label='Val Loss',   marker='o')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, val_mious, label='Val mIoU', marker='o', color='green')
    axes[1].set_title('Validation mIoU History')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_PATH, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Training history saved to: {plot_path}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
