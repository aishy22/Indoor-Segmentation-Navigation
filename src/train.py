# src/train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Path setup for Colab
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

from config import Config
from data_prep import get_dataloaders
from model import create_model


# ============================================
# FOCAL LOSS
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        ce_loss    = F.cross_entropy(logits, targets,
                                     weight=self.class_weights,
                                     reduction='none')
        pt         = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean(), ce_loss.mean()


# ============================================
# COMBINED FOCAL + DICE LOSS
# ============================================
class CombinedLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal       = FocalLoss(class_weights, gamma)

    def forward(self, logits, targets):
        focal_loss, ce = self.focal(logits, targets)

        probs      = torch.softmax(logits, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)

        dims      = (0, 2, 3)
        intersect = (probs * targets_oh).sum(dims)
        union     = (probs + targets_oh).sum(dims)
        dice      = 1 - (2 * intersect + 1e-6) / (union + 1e-6)
        dice_loss = dice.mean()

        total = focal_loss + self.dice_weight * dice_loss
        return total, ce, dice_loss


# ============================================
# mIoU METRIC — WITH PER-CLASS BREAKDOWN
# ============================================
def compute_miou(preds, targets, num_classes=4):
    ious        = []
    preds_np    = preds.cpu().numpy().flatten()
    targets_np  = targets.cpu().numpy().flatten()
    class_names = ['floor', 'obstacle/wall', 'door', 'no-go']

    print("   Per-class IoU:", end=" ")
    for cls in range(num_classes):
        pred_cls     = (preds_np == cls)
        target_cls   = (targets_np == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union        = np.logical_or(pred_cls,  target_cls).sum()

        if union == 0:
            print(f"{class_names[cls]}=N/A", end="  ")
            continue
        iou = intersection / union
        ious.append(iou)
        print(f"{class_names[cls]}={iou:.4f}", end="  ")
    print()

    return np.mean(ious) if ious else 0.0


# ============================================
# MAIN TRAINING
# ============================================
def main():
    Config.setup()

    device = Config.get_device()
    print(f"\n🔧 Detected device: {device}")
    if device == 'cpu':
        print("⚠️ WARNING: Training on CPU — this will be very slow!")
    else:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    Config.print_config()

    # ── Fresh start — delete old checkpoint ───
    checkpoint_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n🗑️  Deleted old checkpoint — fresh start!")
    else:
        print("\n🚀 No checkpoint found — starting fresh.")

    # ── Data ──────────────────────────────────
    print("\n📁 Creating datasets...")
    train_loader, val_loader = get_dataloaders()

    # ── Model ─────────────────────────────────
    model = create_model(pretrained=True).to(device)

    # ── Loss ──────────────────────────────────
    # Best weights from data frequency analysis + tuned door weight
    class_weights = torch.tensor(
        [0.3868, 0.0444, 10.0, 1.5]
    ).to(device)
    criterion = CombinedLoss(
        class_weights=class_weights,
        gamma=2.0,
        dice_weight=0.5
    ).to(device)

    print("\n✅ Loss: CombinedLoss (FocalLoss + DiceLoss)")
    print(f"   Class weights : {class_weights.tolist()}")
    print(f"   Gamma         : 2.0")
    print(f"   Dice weight   : 0.5")

    # ── Optimizer ─────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,           # full LR for fresh training
        weight_decay=1e-4
    )

    # ── Scheduler — tracks mIoU now ───────────
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',   # higher mIoU = better
        factor=0.5, patience=4,  # more patient than before
    )

    # ── Training state ────────────────────────
    start_epoch = 1
    best_miou   = 0.0
    TOTAL_EPOCHS = 40          # 40 epochs for full clean run

    train_losses, val_losses, val_mious = [], [], []

    print("\n" + "=" * 60)
    print("🚀 STARTING FULL CLEAN TRAINING (40 epochs)")
    print(f"   Training batches  : {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Epochs            : {start_epoch} → {TOTAL_EPOCHS}")
    print(f"   LR                : 1e-4 (full)")
    print(f"   Device            : {device}")
    print("=" * 60)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):

        # ── Train ───────────────────────────────
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{TOTAL_EPOCHS} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            outputs         = model(images)
            total, ce, dice = criterion(outputs, masks)
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
        val_loss    = 0.0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                      desc=f"Epoch {epoch}/{TOTAL_EPOCHS} [Val]"):
                images = images.to(device)
                masks  = masks.to(device)

                outputs         = model(images)
                total, ce, dice = criterion(outputs, masks)
                val_loss       += total.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_targets.append(masks)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        all_preds   = torch.cat(all_preds,   dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        miou        = compute_miou(all_preds, all_targets,
                                   num_classes=Config.NUM_CLASSES)
        val_mious.append(miou)

        scheduler.step(miou)

        print(
            f"\nEpoch {epoch:02d}/{TOTAL_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val mIoU: {miou:.4f}"
        )

        # ── Save best by mIoU ────────────────────
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss'       : avg_val_loss,
                'val_miou'            : miou,
            }, checkpoint_path)
            print(f"  ✓ Saved best model → val_loss: {avg_val_loss:.4f} | mIoU: {miou:.4f}")

    print(f"\n✅ Training complete!")
    print(f"   Best mIoU : {best_miou:.4f}")

    # ── Plot Training History ──────────────────
    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, train_losses, label='Train Loss', marker='o')
    axes[0].plot(epochs_range, val_losses,   label='Val Loss',   marker='o')
    axes[0].set_title('Loss History (Full Clean Run)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, val_mious, label='Val mIoU',
                 marker='o', color='green')
    axes[1].set_title('Validation mIoU (Full Clean Run)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_PATH, 'training_history_final.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Training history saved to: {plot_path}")


if __name__ == "__main__":
    main()
