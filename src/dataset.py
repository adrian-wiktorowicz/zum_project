# dataset classes

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from . import config
from .audio_utils import preprocess_audio, extract_rf_features, SpecAugment


class EDMDataset(Dataset):
    """Dataset for CNN - returns mel spectrograms"""
    
    def __init__(self, manifest_df, split='train', augment=False, return_rf_features=False):
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.split = split
        self.augment = augment and (split == 'train')
        self.return_rf_features = return_rf_features
        
        self.mode = 'random' if split == 'train' else 'center'
        self.augmenter = SpecAugment() if self.augment else None
        
        # filter missing files
        valid_mask = self.df['filepath'].apply(lambda x: Path(x).exists() if pd.notna(x) and x else False)
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} files missing in {split}")
            self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"{split}: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label_idx = int(row['label_idx'])
        
        mel = preprocess_audio(
            filepath,
            mode=self.mode,
            augment=self.augment,
            augmenter=self.augmenter
        )
        
        if self.return_rf_features:
            features = extract_rf_features(mel)
            return torch.from_numpy(features), label_idx
        else:
            return mel, label_idx


class RFDataset:
    """Loads all features into memory for sklearn"""
    
    def __init__(self, manifest_df, split='train'):
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.split = split
        
        # filter valid
        valid_mask = self.df['filepath'].apply(lambda x: Path(x).exists() if pd.notna(x) and x else False)
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"Loading {split} features ({len(self.df)} samples)...")
        
        self.features = []
        self.labels = []
        
        from tqdm import tqdm
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                mel = preprocess_audio(row['filepath'], mode='center', augment=False)
                feat = extract_rf_features(mel)
                self.features.append(feat)
                self.labels.append(int(row['label_idx']))
            except Exception as e:
                print(f"Error: {row['filepath']}: {e}")
        
        self.X = np.array(self.features)
        self.y = np.array(self.labels)
        
        print(f"Loaded {len(self.X)} samples, shape {self.X.shape}")
    
    def get_data(self):
        return self.X, self.y


class ASTDataset(Dataset):
    """Dataset for AST transformer"""
    
    def __init__(self, manifest_df, split='train', model_name=config.AST_MODEL_NAME):
        from transformers import ASTFeatureExtractor
        
        self.df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.split = split
        
        # filter missing
        valid_mask = self.df['filepath'].apply(lambda x: Path(x).exists() if pd.notna(x) and x else False)
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} files missing in {split}")
            self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"Loading feature extractor: {model_name}")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.target_sr = self.feature_extractor.sampling_rate
        
        print(f"AST {split}: {len(self.df)} samples, sr={self.target_sr}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label_idx = int(row['label_idx'])
        
        import librosa
        waveform, _ = librosa.load(filepath, sr=self.target_sr, mono=True)
        
        # crop/pad to 10s
        target_length = self.target_sr * config.CLIP_DURATION_SEC
        if len(waveform) > target_length:
            start = (len(waveform) - target_length) // 2
            waveform = waveform[start:start + target_length]
        elif len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        
        input_values = inputs.input_values.squeeze(0)
        return input_values, label_idx


def create_ast_dataloaders(manifest_df, batch_size=config.AST_BATCH_SIZE, 
                            model_name=config.AST_MODEL_NAME, num_workers=0):
    """helper for AST dataloaders"""
    train_dataset = ASTDataset(manifest_df, split='train', model_name=model_name)
    val_dataset = ASTDataset(manifest_df, split='val', model_name=model_name)
    test_dataset = ASTDataset(manifest_df, split='test', model_name=model_name)
    
    loader_kwargs = {'pin_memory': True}
    if num_workers > 0:
        loader_kwargs['num_workers'] = num_workers
        loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader


def create_dataloaders(manifest_df, batch_size=config.CNN_BATCH_SIZE, augment=True, num_workers=6):
    """helper for CNN dataloaders"""
    train_dataset = EDMDataset(manifest_df, split='train', augment=augment)
    val_dataset = EDMDataset(manifest_df, split='val', augment=False)
    test_dataset = EDMDataset(manifest_df, split='test', augment=False)
    
    loader_kwargs = {
        'pin_memory': True,
        'num_workers': num_workers,
    }
    
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader
