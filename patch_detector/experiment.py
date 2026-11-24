"""Train, resume, or evaluate the HiDF patch detector."""
import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageOps
import sklearn
from sklearn import metrics
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data import read_manifest
from model import PatchDetector


def jpeg(image, quality):
    with io.BytesIO() as buffer:
        image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")


class Faces(Dataset):
    def __init__(self, root, rows, training=False, quality=None):
        self.root, self.rows = root, rows
        self.training, self.quality = training, quality
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        key, label, _ = self.rows[index]
        with Image.open(self.root / key) as file:
            image = file.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            return self.transform(image), self.transform(jpeg(image, random.randint(50, 95))), label
        if self.quality is not None:
            image = jpeg(image, self.quality)
        return self.transform(image), label


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    truth, scores = [], []
    for images, labels in loader:
        scores.extend(model(images.to(device)).sigmoid().cpu().tolist())
        truth.extend(labels.tolist())
    predicted = np.asarray(scores) > 0.5
    report = {
        "accuracy": metrics.accuracy_score(truth, predicted),
        "precision": metrics.precision_score(truth, predicted, zero_division=0),
        "recall": metrics.recall_score(truth, predicted, zero_division=0),
        "f1": metrics.f1_score(truth, predicted, zero_division=0),
        "auc": metrics.roc_auc_score(truth, scores),
        "average_precision": metrics.average_precision_score(truth, scores),
    }
    return {k: float(v) for k, v in report.items()}, scores


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--variant", choices=["global", "patch", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="Disable ImageNet initialization")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive and workers non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = read_manifest(args.data)
    digest = hashlib.sha256((args.data / "hidf_clean_source_manifest.csv").read_bytes()).hexdigest()
    run = args.runs / f"{args.variant}-seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    best, last = run / "best.pt", run / "last.pt"
    settings = dict(variant=args.variant, seed=args.seed, scratch=args.scratch, batch=args.batch,
                    workers=args.workers, manifest_sha256=digest, learning_rate=0.0001,
                    consistency_weight=0.5 if args.variant == "full" else 0.0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoint = best if args.mode == "test" else last
    state = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
    if args.mode == "test" and state is None:
        parser.error(f"Train the model first. Missing {best}")
    if state and (state["settings"]["manifest_sha256"] != digest or
                  (args.mode == "train" and state["settings"] != settings)):
        parser.error("Manifest or training settings changed. Select the original settings or a new runs directory.")
    model = PatchDetector(args.variant != "global", state is None and not args.scratch).to(device)
    if state:
        model.load_state_dict(state["model"])

    def loader(split, quality=None, epoch=0):
        training = split == "train"
        generator = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(Faces(args.data, splits[split], training, quality), batch_size=args.batch,
                          shuffle=training, num_workers=args.workers, generator=generator,
                          pin_memory=device.type == "cuda")

    if args.mode == "test":
        table = {}
        for condition, quality in [("clean", None), ("Q_90", 90), ("Q_70", 70), ("Q_50", 50)]:
            table[condition], scores = evaluate(model, loader("test", quality), device)
            with (run / f"predictions_{condition}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["image", "label", "source_group", "p_fake"])
                writer.writerows((*row, score) for row, score in zip(splits["test"], scores))
        report = dict(epoch=state["epoch"], settings=state["settings"], metrics=table)
        (run / "test.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(table, indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    history = []
    if state:
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
    info = dict(settings=settings, torch=str(torch.__version__), torchvision=torchvision.__version__,
                pillow=PIL.__version__, sklearn=sklearn.__version__, numpy=np.__version__,
                device=torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
                parameters=sum(p.numel() for p in model.parameters()), split_sizes={k: len(v) for k,v in splits.items()})
    (run / "run.json").write_text(json.dumps(info, indent=2))
    best_auc = max((row["auc"] for row in history), default=-1)
    for epoch in range(len(history) + 1, args.epochs + 1):
        model.train()
        random.seed(args.seed + epoch)
        total_loss = 0
        for clean, compressed, labels in loader("train", epoch=epoch):
            images = torch.cat([clean, compressed]).to(device)
            targets = labels.to(device).float()
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets.repeat(2))
            if args.variant == "full":
                first, second = logits.sigmoid().chunk(2)
                loss = loss + settings["consistency_weight"] * F.mse_loss(first, second)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        val, _ = evaluate(model, loader("val"), device)
        history.append(dict(epoch=epoch, train_loss=total_loss / len(splits["train"]), **val))
        print(json.dumps(history[-1]), flush=True)
        snapshot = dict(settings=settings, epoch=epoch, model=model.state_dict())
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            save_checkpoint(best, snapshot)
        save_checkpoint(last, dict(**snapshot, optimizer=optimizer.state_dict(), history=history))
        (run / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
