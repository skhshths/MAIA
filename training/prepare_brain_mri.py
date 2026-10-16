"""
Builds the brain MRI training data for MAIA's two brain tumor models.

Source: Sartaj Bhuvaji et al., "Brain Tumor Classification (MRI)" dataset
        https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet
        (also on Kaggle as sartajbhuvaji/brain-tumor-classification-mri)

It writes two ImageFolder-style datasets:

    data/tumor_classification/{train,val,test}/{glioma,meningioma,none,pituitary}
    data/tumor_detection/{train,val,test}/{no,yes}

Class folder names match models/<model>/labels.txt so the exported ONNX
models drop straight into app.py.

Duplicates are a real problem in this dataset. About three quarters of the
official "Testing" images are byte-identical or pixel-identical copies of
"Training" images (including every no_tumor test image), so scores on the
official split are inflated. Instead we:
  1. pool Training + Testing
  2. hash every image (md5 + a difference hash that survives resizing) and
     keep one copy of each duplicate group
  3. make a fresh stratified train / val / test split

Usage:
    python training/prepare_brain_mri.py --out data
    python training/prepare_brain_mri.py --src path/to/existing/clone --out data
"""
import argparse
import hashlib
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

REPO = "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet"

# dataset folder -> MAIA label
CLASS_MAP = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "no_tumor": "none",
    "pituitary_tumor": "pituitary",
}
DETECTION_MAP = {"glioma": "yes", "meningioma": "yes", "pituitary": "yes", "none": "no"}


def dhash(path, size=16):
    """Difference hash: survives resizing / recompression, so it catches near-duplicates."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits).tobytes()


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect(split_dir):
    items = []
    for folder, label in CLASS_MAP.items():
        d = os.path.join(split_dir, folder)
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, name), label))
    return items


def dedupe(items):
    """Keeps the first copy of every image; drops exact (md5) and near (dhash) duplicates."""
    seen_md5, seen_hash = {}, {}
    kept, dropped, conflicts = [], Counter(), 0
    for path, label in items:
        m, h = md5(path), dhash(path)
        first = seen_md5.get(m) or seen_hash.get(h)
        if first is not None:
            dropped["exact" if m in seen_md5 else "near"] += 1
            if first != label:
                conflicts += 1
            continue
        seen_md5[m] = seen_hash[h] = label
        kept.append((path, label))
    return kept, dropped, conflicts


def write_split(items, out_root, split, label_fn):
    for i, (path, label) in enumerate(items):
        dst_dir = os.path.join(out_root, split, label_fn(label))
        os.makedirs(dst_dir, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        shutil.copyfile(path, os.path.join(dst_dir, f"{i:05d}{ext}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="existing clone of the dataset repo (cloned if missing)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = args.src or os.path.join(args.out, "_raw", "sartaj")
    if not os.path.isdir(os.path.join(src, "Training")):
        os.makedirs(os.path.dirname(src), exist_ok=True)
        print(f"cloning {REPO} ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO, src], check=True)

    items = collect(os.path.join(src, "Training")) + collect(os.path.join(src, "Testing"))
    print(f"raw: {len(items)} images")

    items, dropped, conflicts = dedupe(items)
    print(f"after dedupe: {len(items)} unique (dropped {dict(dropped)}, {conflicts} with a different label)")

    # Fresh stratified train / val / test split
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for it in items:
        by_label[it[1]].append(it)
    train, val, test = [], [], []
    for label, its in sorted(by_label.items()):
        rng.shuffle(its)
        n_test = round(len(its) * args.test_frac)
        n_val = round(len(its) * args.val_frac)
        test += its[:n_test]
        val += its[n_test:n_test + n_val]
        train += its[n_test + n_val:]

    for task, fn in [("tumor_classification", lambda l: l), ("tumor_detection", lambda l: DETECTION_MAP[l])]:
        root = os.path.join(args.out, task)
        shutil.rmtree(root, ignore_errors=True)
        for split, items in [("train", train), ("val", val), ("test", test)]:
            write_split(items, root, split, fn)
            counts = Counter(fn(l) for _, l in items)
            print(f"{task:22s} {split:5s} {sum(counts.values()):5d}  {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
