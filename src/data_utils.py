# data download and preprocessing utils

import os
import re
import time
import json
import requests
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from tqdm import tqdm

from . import config


def download_file(url, dest_path, desc="Downloading"):
    """download with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_mtg_metadata(output_dir=None):
    """download metadata from github"""
    output_dir = output_dir or config.RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    genre_path = output_dir / "autotagging_genre.tsv"
    meta_path = output_dir / "raw.meta.tsv"
    
    if not genre_path.exists():
        print("Downloading autotagging_genre.tsv...")
        download_file(config.MTG_GENRE_TSV_URL, genre_path, "Genre TSV")
    else:
        print(f"Genre TSV already exists: {genre_path}")
    
    if not meta_path.exists():
        print("Downloading raw.meta.tsv...")
        download_file(config.MTG_META_TSV_URL, meta_path, "Meta TSV")
    else:
        print(f"Meta TSV already exists: {meta_path}")
    
    return genre_path, meta_path


def load_genre_metadata(path):
    """parse autotagging_genre.tsv"""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                track_id = parts[0]
                artist_id = parts[1]
                album_id = parts[2]
                path = parts[3]
                duration = float(parts[4])
                tags = parts[5:]
                rows.append({
                    'track_id': track_id,
                    'artist_id': artist_id,
                    'album_id': album_id,
                    'path': path,
                    'duration': duration,
                    'tags': tags
                })
    
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} tracks from genre metadata")
    return df


def load_meta_tsv(path):
    """parse raw.meta.tsv"""
    df = pd.read_csv(path, sep='\t', encoding='utf-8')
    df.columns = [c.lower() for c in df.columns]
    print(f"Loaded {len(df)} tracks from meta TSV")
    return df


def normalize_tag(tag):
    """lowercase, strip, remove prefix"""
    tag = re.sub(r'^genre---', '', tag, flags=re.IGNORECASE)
    tag = tag.lower().strip()
    tag = tag.replace('_', ' ').replace('-', ' ')
    tag = re.sub(r'\s+', ' ', tag)
    return tag


def match_target_labels(tags):
    """match tags to our 4 classes"""
    matched = set()
    
    for tag in tags:
        normalized = normalize_tag(tag)
        
        for target_label, synonyms in config.TAG_SYNONYMS.items():
            for synonym in synonyms:
                if normalized == synonym or normalized == synonym.replace(' ', ''):
                    matched.add(target_label)
                    break
    
    return list(matched)


def build_labeled_dataset(genre_df, max_per_class=config.MAX_PER_CLASS,
                          min_per_class=config.MIN_PER_CLASS, seed=config.RANDOM_SEED,
                          allow_low_counts=False):
    """filter to single-label tracks and balance classes"""
    
    genre_df = genre_df.copy()
    genre_df['matched_labels'] = genre_df['tags'].apply(match_target_labels)
    genre_df['num_matched'] = genre_df['matched_labels'].apply(len)
    
    # keep only single-label
    single_label_df = genre_df[genre_df['num_matched'] == 1].copy()
    single_label_df['label'] = single_label_df['matched_labels'].apply(lambda x: x[0])
    
    print(f"\nFiltered to {len(single_label_df)} single-label tracks")
    
    class_counts = single_label_df['label'].value_counts()
    print("\nClass distribution before capping:")
    for label in config.TARGET_CLASSES:
        count = class_counts.get(label, 0)
        print(f"  {label}: {count}")
    
    # check min counts
    low_count_classes = []
    for label in config.TARGET_CLASSES:
        count = class_counts.get(label, 0)
        if count < min_per_class:
            low_count_classes.append((label, count))
    
    if low_count_classes and not allow_low_counts:
        print(f"\n ERROR: Classes with insufficient data:")
        for label, count in low_count_classes:
            print(f"  {label}: {count} (need at least {min_per_class})")
        raise ValueError(f"Insufficient data for classes: {low_count_classes}")
    
    # cap at max_per_class
    balanced_dfs = []
    for label in config.TARGET_CLASSES:
        label_df = single_label_df[single_label_df['label'] == label]
        if len(label_df) > max_per_class:
            label_df = label_df.sample(n=max_per_class, random_state=seed)
        balanced_dfs.append(label_df)
    
    result_df = pd.concat(balanced_dfs, ignore_index=True)
    
    result_df['label_idx'] = result_df['label'].map(config.CLASS_TO_IDX)
    result_df['tags_raw'] = result_df['tags'].apply(lambda x: ';'.join(x))
    
    print(f"\nFinal class distribution:")
    final_counts = result_df['label'].value_counts()
    for label in config.TARGET_CLASSES:
        print(f"  {label}: {final_counts.get(label, 0)}")
    print(f"\nTotal: {len(result_df)} tracks")
    
    return result_df[['track_id', 'artist_id', 'album_id', 'duration', 'label', 'label_idx', 'tags_raw']]


def artist_disjoint_split(df, train_ratio=config.TRAIN_RATIO, 
                          val_ratio=config.VAL_RATIO, seed=config.RANDOM_SEED):
    """split data so no artist appears in multiple splits"""
    import numpy as np
    np.random.seed(seed)
    
    artists = df['artist_id'].unique()
    np.random.shuffle(artists)
    
    n_artists = len(artists)
    train_end = int(n_artists * train_ratio)
    val_end = int(n_artists * (train_ratio + val_ratio))
    
    train_artists = set(artists[:train_end])
    val_artists = set(artists[train_end:val_end])
    test_artists = set(artists[val_end:])
    
    def get_split(artist_id):
        if artist_id in train_artists:
            return 'train'
        elif artist_id in val_artists:
            return 'val'
        else:
            return 'test'
    
    df = df.copy()
    df['split'] = df['artist_id'].apply(get_split)
    
    print("\nSplit statistics:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        n_tracks = len(split_df)
        n_artists = split_df['artist_id'].nunique()
        print(f"  {split}: {n_tracks} tracks, {n_artists} artists")
        for label in config.TARGET_CLASSES:
            label_count = len(split_df[split_df['label'] == label])
            print(f"    {label}: {label_count}")
    
    return df


class JamendoClient:
    """client for downloading from jamendo api"""
    
    def __init__(self, client_id=config.JAMENDO_CLIENT_ID, rate_limit=config.JAMENDO_RATE_LIMIT,
                 max_retries=config.JAMENDO_MAX_RETRIES, retry_backoff=config.JAMENDO_RETRY_BACKOFF):
        self.client_id = client_id
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.last_request_time = 0
        
    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        wait_time = (1.0 / self.rate_limit) - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _extract_numeric_id(self, track_id):
        """track_0000214 -> 214"""
        match = re.search(r'track_0*(\d+)', track_id)
        if match:
            return match.group(1)
        return str(int(track_id))
    
    def get_track_info(self, track_id):
        """get track info from api"""
        numeric_id = self._extract_numeric_id(track_id)
        
        params = {
            'client_id': self.client_id,
            'format': 'json',
            'id': numeric_id
        }
        
        url = f"{config.JAMENDO_API_BASE}/tracks/?{urlencode(params)}"
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                response = requests.get(url, timeout=30)
                
                if response.status_code == 429:
                    wait = self.retry_backoff ** (attempt + 1)
                    print(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                if response.status_code >= 500:
                    wait = self.retry_backoff ** (attempt + 1)
                    time.sleep(wait)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    return data['results'][0]
                else:
                    return None
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_backoff ** (attempt + 1)
                    time.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def download_audio(self, track_id, dest_path, skip_existing=True):
        """download audio preview"""
        if skip_existing and dest_path.exists():
            return True, "already_exists"
        
        track_info = self.get_track_info(track_id)
        
        if not track_info:
            return False, "track_not_found"
        
        audio_url = track_info.get('audio')
        if not audio_url:
            return False, "no_audio_url"
        
        try:
            self._rate_limit_wait()
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            
            return True, "downloaded"
            
        except Exception as e:
            return False, f"download_error: {e}"


def smoke_test_jamendo(track_ids, client=None):
    """quick test to check if jamendo api works"""
    client = client or JamendoClient()
    
    print("\n" + "="*60)
    print("JAMENDO API SMOKE TEST")
    print("="*60)
    
    all_passed = True
    
    for track_id in track_ids:
        print(f"\nTesting track_id: {track_id}")
        
        track_info = client.get_track_info(track_id)
        
        if not track_info:
            print(f"  FAILED: Could not get track info")
            all_passed = False
            continue
        
        audio_url = track_info.get('audio')
        if not audio_url:
            print(f"  FAILED: No audio URL")
            all_passed = False
            continue
        
        try:
            response = requests.head(audio_url, timeout=10)
            if response.status_code == 200:
                print(f"  OK: Audio URL accessible")
                print(f"    Name: {track_info.get('name', 'N/A')}")
            else:
                print(f"  FAILED: Audio URL returned {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"  FAILED: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_passed


def download_audio_batch(manifest_df, output_dir=None, client=None):
    """download all audio files"""
    output_dir = output_dir or config.AUDIO_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = client or JamendoClient()
    
    manifest_df = manifest_df.copy()
    manifest_df['filepath'] = ''
    
    stats = {'total': len(manifest_df), 'ok': 0, 'failed': 0, 'skipped': 0}
    failures = []
    
    print(f"\nDownloading {len(manifest_df)} audio files...")
    
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df)):
        track_id = row['track_id']
        numeric_id = client._extract_numeric_id(track_id)
        dest_path = output_dir / f"{numeric_id}.mp3"
        
        success, message = client.download_audio(track_id, dest_path)
        
        if success:
            manifest_df.at[idx, 'filepath'] = str(dest_path.resolve())
            if message == "already_exists":
                stats['skipped'] += 1
            else:
                stats['ok'] += 1
        else:
            stats['failed'] += 1
            failures.append({'track_id': track_id, 'reason': message})
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Downloaded: {stats['ok']}")
    print(f"Skipped (cached): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    if failures:
        print("\nFailure reasons:")
        reason_counts = {}
        for f in failures:
            reason = f['reason'].split(':')[0]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
    
    print("="*60 + "\n")
    
    return manifest_df


def save_manifest(df, path=None):
    path = path or (config.PROCESSED_DATA_DIR / "manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved manifest to: {path}")


def load_manifest(path=None):
    path = path or (config.PROCESSED_DATA_DIR / "manifest.csv")
    df = pd.read_csv(path)
    
    # fix relative paths
    if 'filepath' in df.columns:
        def fix_path(p):
            if pd.isna(p) or not p:
                return p
            resolved = Path(p).resolve()
            return str(resolved)
        df['filepath'] = df['filepath'].apply(fix_path)
    
    return df
