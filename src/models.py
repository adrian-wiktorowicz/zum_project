"""
Models for EDM classification
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class ConvBlock(nn.Module):
    """single conv block: conv -> bn -> relu -> pool -> dropout"""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SimpleCNN(nn.Module):
    """
    4-block CNN for audio classification.
    Input: [B, 1, 128, T] mel spectrogram
    """
    
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        
        self.num_classes = num_classes
        
        # conv blocks
        self.block1 = ConvBlock(1, 16, dropout=0.1)
        self.block2 = ConvBlock(16, 32, dropout=0.2)
        self.block3 = ConvBlock(32, 64, dropout=0.3)
        self.block4 = ConvBlock(64, 128, dropout=0.4)
        
        # classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RandomForestModel:
    """Wrapper for sklearn RandomForest"""
    
    def __init__(
        self,
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.RANDOM_SEED
    ):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        print(f"Training RF on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("Done.")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"Saved to {path}")
    
    @classmethod
    def load(cls, path):
        import joblib
        instance = cls()
        instance.model = joblib.load(path)
        instance.is_fitted = True
        return instance


class ASTModel(nn.Module):
    """Audio Spectrogram Transformer wrapper for fine-tuning"""
    
    def __init__(
        self,
        model_name=config.AST_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        freeze_encoder=True,
        unfreeze_last_n=2
    ):
        super().__init__()
        
        from transformers import ASTModel as HFASTModel, ASTFeatureExtractor
        
        self.num_classes = num_classes
        
        # load pretrained
        print(f"Loading AST: {model_name}")
        self.ast = HFASTModel.from_pretrained(model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        
        self.hidden_size = self.ast.config.hidden_size
        
        # new classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, num_classes)
        )
        
        # freeze most of encoder
        if freeze_encoder:
            for param in self.ast.parameters():
                param.requires_grad = False
            
            # unfreeze last N layers
            if unfreeze_last_n > 0:
                encoder_layers = self.ast.encoder.layer
                for layer in encoder_layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        print(f"Trainable params: {self.count_parameters()}")
    
    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        
        # use CLS token
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states[:, 0, :]
        
        logits = self.classifier(pooled)
        return logits
    
    def preprocess(self, waveform, sample_rate=config.SAMPLE_RATE):
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        return inputs.input_values.squeeze(0)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
