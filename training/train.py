"""
Fine-tunes a ResNet-18 for one MAIA task and exports it to ONNX.

The exported model has the same contract as the original jetson-inference
models, so it drops straight into models/<task>/ without touching app.py:
    input  "input_0"  float32 [1, 3, 224, 224], ImageNet-normalised RGB
    output "output_0" float32 [1, n_classes], softmax probabilities

Expects an ImageFolder layout (see prepare_brain_mri.py):
    <data>/train/<label>/*.jpg
    <data>/val/<label>/*.jpg

Starting weights: ImageNet ResNet-18 from timm's "ResNet strikes back" release
(resnet18_a1_0-d63eafa0.pth, published on the timm GitHub releases page).
Pass --weights to point at a local copy.

Example:
    python training/train.py --data data/tumor_classification \
        --labels models/tumor_classification/labels.txt \
        --out runs/tumor_classification --epochs 15
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

WEIGHTS_URL = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Folder(Dataset):
    """Like torchvision's ImageFolder, but class order comes from labels.txt."""

    def __init__(self, root, labels, transform):
        self.items = []
        for idx, label in enumerate(labels):
            d = os.path.join(root, label)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                self.items.append((os.path.join(d, name), idx))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), y


def build_model(n_classes, weights_path):
    model = models.resnet18(num_classes=1000)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def evaluate(model, loader, n_classes):
    model.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss_sum += nn.functional.cross_entropy(out, y, reduction="sum").item()
            n += len(y)
            for t, p in zip(y.tolist(), out.argmax(1).tolist()):
                conf[t, p] += 1
    recall = conf.diagonal() / np.maximum(conf.sum(1), 1)
    return {
        "loss": loss_sum / max(n, 1),
        "acc": conf.diagonal().sum() / max(conf.sum(), 1),
        "balanced_acc": recall[conf.sum(1) > 0].mean(),
        "confusion": conf.tolist(),
    }


def export_onnx(model, path):
    model.eval()
    wrapped = nn.Sequential(model, nn.Softmax(dim=1))
    dummy = torch.randn(1, 3, 224, 224)
    kwargs = dict(input_names=["input_0"], output_names=["output_0"], opset_version=13)
    try:
        torch.onnx.export(wrapped, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # older torch without the dynamo flag
        torch.onnx.export(wrapped, dummy, path, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="ImageNet ResNet-18 .pth (downloaded if omitted)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    with open(args.labels) as f:
        labels = [line.strip() for line in f if line.strip()]

    weights = args.weights
    if not weights:
        weights = os.path.join(args.out, os.path.basename(WEIGHTS_URL))
        if not os.path.exists(weights):
            torch.hub.download_url_to_file(WEIGHTS_URL, weights)

    # Eval resize matches app.py exactly: straight resize to 224x224, no crop
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Mild augmentation: scans are fairly standardised, so nothing too wild
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = Folder(os.path.join(args.data, "train"), labels, train_tf)
    val_ds = Folder(os.path.join(args.data, "val"), labels, eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=args.workers)

    # Inverse-frequency class weights so small classes (e.g. "no tumor") aren't ignored
    counts = np.bincount([y for _, y in train_ds.items], minlength=len(labels)).astype(np.float32)
    class_w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)))
    print(f"train {len(train_ds)}  val {len(val_ds)}  counts {dict(zip(labels, counts.astype(int).tolist()))}")

    model = build_model(len(labels), weights)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val = evaluate(model, val_dl, len(labels))
        row = {"epoch": epoch, "train_loss": running / len(train_dl), "val_loss": val["loss"],
               "val_acc": float(val["acc"]), "val_balanced_acc": float(val["balanced_acc"]),
               "secs": round(time.time() - t0)}
        log.append(row)
        print(f"epoch {epoch:2d}  train_loss {row['train_loss']:.3f}  val_loss {row['val_loss']:.3f}  "
              f"val_acc {row['val_acc']:.3f}  val_bal_acc {row['val_balanced_acc']:.3f}  ({row['secs']}s)", flush=True)

        if val["balanced_acc"] > best:
            best = val["balanced_acc"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt")))
    export_onnx(model, os.path.join(args.out, "resnet18.onnx"))
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump({"labels": labels, "best_val_balanced_acc": best, "epochs": log, "args": vars(args)}, f, indent=2)
    print(f"done. best val balanced acc {best:.3f} -> {os.path.join(args.out, 'resnet18.onnx')}")


if __name__ == "__main__":
    main()
