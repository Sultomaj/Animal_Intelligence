import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curves(history, name, save_dir):
    """Plots and saves the training and validation learning curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker='o')
    plt.title(f"{name} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker='o')
    plt.title(f"{name} - Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_learning_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_custom(model, loader, device, class_names, name, save_dir):
    """Generates and saves the confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Hide values if there are too many classes to prevent text overlapping
    show_values = len(class_names) <= 20
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=90, include_values=show_values)
    
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate per-class accuracy for the JSON report
    class_acc = cm.diagonal() / cm.sum(axis=1)
    class_acc_dict = {class_names[i]: round(acc * 100, 2) if not np.isnan(acc) else 0.0 for i, acc in enumerate(class_acc)}
    return class_acc_dict

def plot_prediction_samples(model, loader, device, class_names, name, save_dir, num_images=10):
    """Saves a grid of 10 random predictions."""
    from torch.utils.data import DataLoader
    
    model.eval()
    temp_loader = DataLoader(loader.dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(temp_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

    # Standard ImageNet denormalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat[:num_images]):
        if i >= len(images): break
            
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)

        pred_label = predictions[i].item()
        true_label = labels[i].item()

        ax.imshow(img)
        title_color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Pred: {class_names[pred_label]}\nTrue: {class_names[true_label]}", color=title_color)
        ax.axis("off")

    plt.suptitle(f"{name} - Prediction Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_prediction_samples.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_all(models_dict, val_loader, class_names, device, save_dir="checkpoints"):
    """
    Main entry point called by train.py.
    Iterates over all models and generates reports.
    """
    print(f"\n{'='*50}\nGenerating Metrics & Plots...\n{'='*50}")
    
    for name, (model, history) in models_dict.items():
        print(f"Processing reports for: {name}")
        
        # 1. Plot Learning Curves
        plot_learning_curves(history, name, save_dir)
        
        # 2. Plot Confusion Matrix and get per-class accuracy
        class_acc_dict = plot_confusion_matrix_custom(model, val_loader, device, class_names, name, save_dir)
        
        # 3. Plot Image Samples
        plot_prediction_samples(model, val_loader, device, class_names, name, save_dir)
        
        # 4. Save JSON Report
        final_metrics = {
            "model_name": name,
            "final_val_loss": round(history["val_loss"][-1], 4),
            "final_val_acc": round(history["val_acc"][-1], 2),
            "best_val_acc": round(max(history["val_acc"]), 2),
            "per_class_accuracy": class_acc_dict
        }
        
        json_path = os.path.join(save_dir, f"{name}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
    print(f"\nAll metrics and visualizations successfully saved to {save_dir}/")