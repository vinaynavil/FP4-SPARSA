# train.py
# Trains ResNet-20 on CIFAR-10 (FP32) and saves checkpoint.
# Target: ~91% test accuracy in 200 epochs.
# RTX 3050 6GB: ~15-20 min total.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cifar10_loader import load_cifar10
from models.resnet20 import ResNet20

# ── Config ────────────────────────────────────────────────────
EPOCHS     = 200
BATCH_SIZE = 128
LR         = 0.1
CHECKPOINT = "checkpoints/resnet20_fp32.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data augmentation (manual, no torchvision needed) ─────────
def augment_batch(x):
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[3])
    # Random crop: pad 4 each side then crop 32×32
    pad = 4
    x = F.pad(x, [pad]*4, mode='reflect')
    i = torch.randint(0, 2*pad, (1,)).item()
    j = torch.randint(0, 2*pad, (1,)).item()
    x = x[:, :, i:i+32, j:j+32]
    return x

import torch.nn.functional as F

def get_loaders(tr_img, tr_lbl, te_img, te_lbl):
    tr_x = torch.tensor(tr_img)
    tr_y = torch.tensor(tr_lbl, dtype=torch.long)
    te_x = torch.tensor(te_img)
    te_y = torch.tensor(te_lbl, dtype=torch.long)

    train_ds = TensorDataset(tr_x, tr_y)
    test_ds  = TensorDataset(te_x, te_y)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

# ── LR schedule: warmup 5 ep, then cosine decay ───────────────
def get_lr(epoch):
    if epoch < 5:
        return LR * (epoch + 1) / 5
    progress = (epoch - 5) / (EPOCHS - 5)
    return LR * 0.5 * (1 + np.cos(np.pi * progress))

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr

# ── Eval ──────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

# ── Train ─────────────────────────────────────────────────────
def train():
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("Loading CIFAR-10...")
    tr_img, tr_lbl, te_img, te_lbl = load_cifar10()
    train_loader, test_loader = get_loaders(tr_img, tr_lbl, te_img, te_lbl)

    model = ResNet20().to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet-20: {total:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(EPOCHS):
        lr = get_lr(epoch)
        set_lr(optimizer, lr)
        model.train()

        total_loss = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = augment_batch(x)            # augment on GPU
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)

        train_acc = 100.0 * correct / total
        train_loss = total_loss / total

        # Evaluate every 10 epochs and at the end
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            test_acc = evaluate(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch'     : epoch + 1,
                    'state_dict': model.state_dict(),
                    'test_acc'  : test_acc,
                }, CHECKPOINT)
                saved = "  ← saved"
            else:
                saved = ""
            print(f"Ep {epoch+1:3d}/{EPOCHS} | lr={lr:.4f} | "
                  f"loss={train_loss:.3f} | train={train_acc:.1f}% | "
                  f"test={test_acc:.1f}%{saved}")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Ep {epoch+1:3d}/{EPOCHS} | lr={lr:.4f} | "
                      f"loss={train_loss:.3f} | train={train_acc:.1f}%")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {CHECKPOINT}")

if __name__ == "__main__":
    train()