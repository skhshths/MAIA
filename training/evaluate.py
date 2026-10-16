"""
Scores one or more ONNX models on a held-out test folder, using exactly the
same preprocessing as app.py. Handy for "is the new model actually better?".

Usage:
    python training/evaluate.py --data data/tumor_classification/test \
        --labels models/tumor_classification/labels.txt \
        --model old=models/tumor_classification/resnet18.onnx \
        --model new=runs/tumor_classification/resnet18.onnx \
        --json results/tumor_classification.json

Add --no-norm-too to also score each model the way MAIA v2.0 did (no
ImageNet normalisation), which shows how much that bug was costing.
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(path, normalize=True):
    # Keep in step with preprocess_image() in app.py
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    if normalize:
        x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None]


def score(model_path, items, labels, normalize=True):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    n = len(labels)
    conf = np.zeros((n, n), dtype=int)
    for path, y in items:
        p = sess.run(None, {name: preprocess(path, normalize)})[0][0]
        conf[y, int(np.argmax(p))] += 1

    support = conf.sum(1)
    recall = conf.diagonal() / np.maximum(support, 1)
    precision = conf.diagonal() / np.maximum(conf.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    present = support > 0
    return {
        "accuracy": float(conf.diagonal().sum() / conf.sum()),
        "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {labels[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])}
                      for i in range(n)},
        "confusion": conf.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test folder with one sub-folder per label")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path/to/model.onnx")
    ap.add_argument("--no-norm-too", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(args.labels) as f:
        labels = [line.strip() for line in f if line.strip()]
    items = []
    for i, label in enumerate(labels):
        d = os.path.join(args.data, label)
        if os.path.isdir(d):
            items += [(os.path.join(d, x), i) for x in sorted(os.listdir(d))]
    print(f"{len(items)} test images, labels {labels}\n")

    results = {}
    for spec in args.model:
        tag, path = spec.split("=", 1)
        runs = [(tag, True)] + ([(f"{tag} (no norm)", False)] if args.no_norm_too else [])
        for name, norm in runs:
            r = score(path, items, labels, norm)
            results[name] = r
            print(f"{name:22s} acc {r['accuracy']:.3f}  bal_acc {r['balanced_accuracy']:.3f}  macro_f1 {r['macro_f1']:.3f}")
            print("    recall  " + "  ".join(f"{k}={v['recall']:.2f}" for k, v in r["per_class"].items()))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"test_images": len(items), "labels": labels, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
