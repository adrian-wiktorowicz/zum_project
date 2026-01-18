# audio loading and preprocessing

import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

from . import config


def load_audio(path, target_sr=config.SAMPLE_RATE, mono=True):
    """load audio file using librosa"""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # using librosa - most compatible with mp3
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    
    # librosa returns [T] for mono, need [1, T]
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    waveform = torch.from_numpy(audio.astype(np.float32))
    
    return waveform, target_sr


def crop_or_pad(waveform, target_length=config.CLIP_SAMPLES, mode='center'):
    """crop or pad to target length"""
    current_length = waveform.shape[-1]
    
    if current_length == target_length:
        return waveform
    
    elif current_length > target_length:
        # crop
        if mode == 'random':
            max_start = current_length - target_length
            start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start = (current_length - target_length) // 2
        
        return waveform[..., start:start + target_length]
    
    else:
        # pad
        pad_total = target_length - current_length
        
        if mode == 'center':
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        else:
            pad_left = 0
            pad_right = pad_total
        
        return torch.nn.functional.pad(waveform, (pad_left, pad_right))


def extract_log_mel(waveform, sample_rate=config.SAMPLE_RATE, n_fft=config.N_FFT,
                    hop_length=config.HOP_LENGTH, n_mels=config.N_MELS):
    """get log-mel spectrogram"""
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        pad_mode='reflect',
        power=2.0
    )
    
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    mel_spec = mel_transform(waveform)
    log_mel = amplitude_to_db(mel_spec)
    
    return log_mel


def normalize_mel(mel):
    """standardize to mean=0, std=1"""
    mean = mel.mean()
    std = mel.std()
    
    if std > 0:
        mel = (mel - mean) / std
    else:
        mel = mel - mean
    
    return mel


class SpecAugment:
    """time and frequency masking"""
    
    def __init__(self, freq_mask_param=config.FREQ_MASK_PARAM, 
                 time_mask_param=config.TIME_MASK_PARAM,
                 num_freq_masks=1, num_time_masks=1):
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, mel):
        for _ in range(self.num_freq_masks):
            mel = self.freq_mask(mel)
        
        for _ in range(self.num_time_masks):
            mel = self.time_mask(mel)
        
        return mel


def preprocess_audio(path, mode='center', augment=False, augmenter=None):
    """full pipeline: load -> crop/pad -> mel -> normalize -> augment"""
    waveform, sr = load_audio(path)
    waveform = crop_or_pad(waveform, config.CLIP_SAMPLES, mode=mode)
    mel = extract_log_mel(waveform, sr)
    mel = normalize_mel(mel)
    
    if augment:
        if augmenter is None:
            augmenter = SpecAugment()
        mel = augmenter(mel)
    
    return mel


def extract_rf_features(mel):
    """extract statistics for random forest (mean, std, etc per mel bin)"""
    from scipy import stats
    
    mel = mel.squeeze(0).numpy()  # [128, T]
    
    features = []
    
    for mel_band in mel:
        features.append(np.mean(mel_band))
        features.append(np.std(mel_band))
        features.append(np.min(mel_band))
        features.append(np.max(mel_band))
        features.append(stats.skew(mel_band))
        features.append(stats.kurtosis(mel_band))
    
    # delta features
    delta = np.diff(mel, axis=1)
    
    for delta_band in delta if delta.shape[1] > 0 else np.zeros_like(mel):
        features.append(np.mean(delta_band))
        features.append(np.std(delta_band))
        features.append(np.min(delta_band))
        features.append(np.max(delta_band))
        features.append(stats.skew(delta_band) if len(delta_band) > 2 else 0)
        features.append(stats.kurtosis(delta_band) if len(delta_band) > 3 else 0)
    
    return np.array(features, dtype=np.float32)
