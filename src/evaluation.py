# evaluation and visualization

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from . import config
from .models import get_device


def compute_metrics(y_true, y_pred, class_names=config.TARGET_CLASSES):
    """compute accuracy, f1, precision, recall"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro')),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
    }
    
    # per-class
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        if i < len(per_class_f1):
            metrics['per_class'][class_name] = {
                'f1': float(per_class_f1[i]),
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i])
            }
    
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=config.TARGET_CLASSES,
                          title="Confusion Matrix", save_path=None, figsize=(8, 6)):
    """plot and optionally save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def plot_learning_curves(history, save_path=None, figsize=(12, 4)):
    """plot loss, accuracy, f1 curves"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # accuracy
    axes[1].plot(epochs, history['val_accuracy'], 'g-', label='Val Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # f1
    axes[2].plot(epochs, history['val_f1'], 'm-', label='Val Macro F1')
    axes[2].set_title('Validation Macro F1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    
    return fig


def plot_model_comparison(results, save_path=None, figsize=(10, 6)):
    """bar chart comparing models"""
    models = list(results.keys())
    metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    
    return fig


def evaluate_model(model, test_loader, output_dir, model_name="CNN", device=None):
    """run inference and compute metrics"""
    device = device or get_device()
    model = model.to(device)
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\nEvaluating {model_name} on test set...")
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device)
            
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['model_name'] = model_name
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS - {model_name}")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print("\nPer-class:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}: P={class_metrics['precision']:.3f} R={class_metrics['recall']:.3f} F1={class_metrics['f1']:.3f}")
    print("=" * 60)
    
    # save
    metrics_path = output_dir / f'metrics_{model_name.lower()}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # confusion matrix
    cm_path = output_dir / f'confusion_matrix_{model_name.lower()}.png'
    plot_confusion_matrix(
        all_labels,
        all_preds,
        title=f"Confusion Matrix - {model_name}",
        save_path=cm_path
    )
    
    return metrics


def evaluate_random_forest(model, X_test, y_test, output_dir):
    """evaluate RF and save results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEvaluating Random Forest...")
    
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics['model_name'] = 'Random Forest'
    
    print("\n" + "=" * 60)
    print("TEST RESULTS - Random Forest")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("=" * 60)
    
    metrics_path = output_dir / 'metrics_rf.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    cm_path = output_dir / 'confusion_matrix_rf.png'
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="Confusion Matrix - Random Forest",
        save_path=cm_path
    )
    
    return metrics


def load_checkpoint_and_evaluate(checkpoint_path, test_loader, output_dir, model_class=None):
    """load saved model and evaluate"""
    from .models import SimpleCNN
    
    model_class = model_class or SimpleCNN
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    model = model_class(num_classes=checkpoint['config']['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return evaluate_model(model, test_loader, output_dir, model_name="CNN")
