"""
train.py — Train MobileNetV2, EfficientNetB0, and a Stacked Ensemble on the
90-class Animal Image Dataset from Kaggle.

After training completes, metrics.py is called automatically to generate all
reports and plots inside the `checkpoints/` folder.

Usage:
    python train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import kagglehub

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NUM_CLASSES   = 90
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LR            = 1e-3
TRAIN_SPLIT   = 0.8
SAVE_DIR      = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
class SubsetDataset(Dataset):
    """Wrapper to safely apply separate transforms to random_split subsets."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def build_dataloaders():
    path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
    dataset_path = os.path.join(path, "animals", "animals")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load raw images (NO transforms yet)
    base_dataset = datasets.ImageFolder(root=dataset_path)
    class_names  = base_dataset.classes

    train_size = int(TRAIN_SPLIT * len(base_dataset))
    val_size   = len(base_dataset) - train_size
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    # Wrap subsets in the custom dataset to safely apply separate transforms
    train_ds = SubsetDataset(train_subset, transform=train_transforms)
    val_ds   = SubsetDataset(val_subset, transform=val_transforms)

    # Note: num_workers=4 removed/reduced for safety, increase if on a strong local machine
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names, base_dataset


def compute_class_weights(base_dataset):
    class_counts = [0] * NUM_CLASSES
    for _, label in base_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    weights = torch.tensor([total / c for c in class_counts], dtype=torch.float)
    return weights.to(DEVICE)

# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), 100 * correct / total


def run_training_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, name):
    print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        print(f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  "
              f"Train Loss={tr_loss:.4f}  Train Acc={tr_acc:.2f}%  "
              f"Val Loss={vl_loss:.4f}  Val Acc={vl_acc:.2f}%")
    return history

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
def build_mobilenet(class_weights):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    for param in model.classifier.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, criterion, optimizer, scheduler


def build_efficientnet(class_weights):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    for param in model.classifier.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, criterion, optimizer, scheduler


class StackedEnsemble(nn.Module):
    def __init__(self, model1, model2, num_classes=NUM_CLASSES):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        for p in self.model1.parameters():
            p.requires_grad = False
        for p in self.model2.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return self.classifier(torch.cat([out1, out2], dim=1))


def build_ensemble(mobilenet, efficientnet):
    model = StackedEnsemble(mobilenet, efficientnet).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, criterion, optimizer, scheduler

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, class_names, base_dataset = build_dataloaders()
    class_weights = compute_class_weights(base_dataset)

    # ── MobileNetV2 ──────────────────────────
    mobilenet, criterion, optimizer, scheduler = build_mobilenet(class_weights)
    mobilenet_history = run_training_loop(
        mobilenet, train_loader, val_loader, criterion, optimizer, scheduler, "MobileNetV2"
    )
    torch.save(mobilenet.state_dict(), os.path.join(SAVE_DIR, "mobilenet_v2_weights.pth"))
    mobilenet.eval()

    # ── EfficientNetB0 ───────────────────────
    efficientnet, criterion, optimizer, scheduler = build_efficientnet(class_weights)
    efficientnet_history = run_training_loop(
        efficientnet, train_loader, val_loader, criterion, optimizer, scheduler, "EfficientNetB0"
    )
    torch.save(efficientnet.state_dict(), os.path.join(SAVE_DIR, "efficientnet_weights.pth"))
    efficientnet.eval()

    # ── Stacked Ensemble ─────────────────────
    ensemble, criterion, optimizer, scheduler = build_ensemble(mobilenet, efficientnet)
    ensemble_history = run_training_loop(
        ensemble, train_loader, val_loader, criterion, optimizer, scheduler, "StackedEnsemble"
    )
    torch.save(ensemble.state_dict(), os.path.join(SAVE_DIR, "stacked_ensemble_weights.pth"))

    print(f"\nAll models saved to {SAVE_DIR}/")

    # ── Auto-run metrics ─────────────────────
    import metrics
    metrics.generate_all(
        models_dict={
            "MobileNetV2":     (mobilenet,    mobilenet_history),
            "EfficientNetB0":  (efficientnet, efficientnet_history),
            "StackedEnsemble": (ensemble,     ensemble_history),
        },
        val_loader=val_loader,
        class_names=class_names,
        device=DEVICE,
        save_dir=SAVE_DIR
    )

if __name__ == "__main__":
    main()