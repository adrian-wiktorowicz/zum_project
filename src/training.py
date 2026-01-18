# training loop and utilities

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .models import SimpleCNN, get_device


class EarlyStopping:
    """stop when val metric stops improving"""
    
    def __init__(self, patience=config.CNN_EARLY_STOPPING_PATIENCE, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Trainer:
    """handles cnn training"""
    
    def __init__(self, model, train_loader, val_loader, output_dir,
                 learning_rate=config.CNN_LEARNING_RATE,
                 weight_decay=config.CNN_WEIGHT_DECAY,
                 epochs=config.CNN_EPOCHS,
                 patience=config.CNN_EARLY_STOPPING_PATIENCE,
                 device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.device = device or get_device()
        
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / n_batches
    
    def validate(self):
        from sklearn.metrics import f1_score
        
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.val_loader, desc="Validating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
                
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        val_loss = total_loss / n_batches
        val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return val_loss, val_accuracy, val_f1
    
    def save_checkpoint(self, filename='best_model.pt'):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'class_to_idx': config.CLASS_TO_IDX,
            'idx_to_class': config.IDX_TO_CLASS,
            'config': {
                'sample_rate': config.SAMPLE_RATE,
                'clip_duration_sec': config.CLIP_DURATION_SEC,
                'n_fft': config.N_FFT,
                'hop_length': config.HOP_LENGTH,
                'n_mels': config.N_MELS,
                'num_classes': config.NUM_CLASSES
            },
            'history': self.history
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def train(self):
        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {self.epochs}")
        print("-" * 60)
        
        best_f1 = 0.0
        
        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            if self.early_stopping(val_f1):
                print(f"  New best! Saving...")
                best_f1 = val_f1
                self.save_checkpoint('best_model.pt')
            else:
                print(f"  No improvement ({self.early_stopping.counter}/{self.early_stopping.patience})")
            
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print("\n" + "=" * 60)
        print(f"Done. Best Val F1: {best_f1:.4f}")
        print("=" * 60)
        
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def train_cnn(train_loader, val_loader, output_dir, **kwargs):
    """convenience function to train CNN"""
    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    trainer = Trainer(model, train_loader, val_loader, output_dir, **kwargs)
    return trainer.train()


def train_random_forest(X_train, y_train, output_dir):
    """train and save RF model"""
    from .models import RandomForestModel
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = RandomForestModel()
    model.fit(X_train, y_train)
    model.save(str(output_dir / 'rf_model.pkl'))
    
    return model
