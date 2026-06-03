# cifar10_loader.py
# Loads CIFAR-10 from local extracted folder, no internet needed.

import pickle
import numpy as np
import os

CIFAR_PATH = "cifar-10-batches-py"

def load_batch(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    images = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(d[b'labels'])
    return images, labels

def load_cifar10():
    train_images, train_labels = [], []
    for i in range(1, 6):
        imgs, lbls = load_batch(os.path.join(CIFAR_PATH, f"data_batch_{i}"))
        train_images.append(imgs)
        train_labels.append(lbls)

    train_images = np.concatenate(train_images)  # (50000, 3, 32, 32)
    train_labels = np.concatenate(train_labels)  # (50000,)

    test_images, test_labels = load_batch(os.path.join(CIFAR_PATH, "test_batch"))

    # Normalize: per-channel mean/std (standard CIFAR-10 values)
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

    train_images = (train_images / 255.0 - mean) / std
    test_images  = (test_images  / 255.0 - mean) / std

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    tr_img, tr_lbl, te_img, te_lbl = load_cifar10()
    print(f"Train: {tr_img.shape}  labels: {tr_lbl.shape}")
    print(f"Test : {te_img.shape}  labels: {te_lbl.shape}")
    print(f"Train image min/max: {tr_img.min():.3f} / {tr_img.max():.3f}")
    print(f"Test  image min/max: {te_img.min():.3f} / {te_img.max():.3f}")
    print("CIFAR-10 loader OK")