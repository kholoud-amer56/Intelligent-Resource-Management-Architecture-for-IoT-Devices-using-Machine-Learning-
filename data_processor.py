# core/data_processor.py
import numpy as np
import pandas as pd
import os

class DataLoader:
    """
    Read CSV, clean columns, keep requested cols, replace non-numeric with 0,
    min-max scale based on train split, build sliding windows and return:
      x_train, x_val, x_test
    (we keep time order: no shuffling -> no leakage)
    """

    def __init__(self, filename, split, cols, verbose=True):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found.")
        df = pd.read_csv(filename, low_memory=False)

        # cleanup column names and BOM
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')

        # check columns
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Columns missing in CSV: {missing}")

        # keep relevant columns
        df = df[cols].copy()

        # replace typical non-numeric placeholders
        df = df.replace('-', 0)
        df = df.fillna(0)

        # ensure numeric
        try:
            self.values = df.values.astype(float)
        except Exception as e:
            raise ValueError(f"Could not convert CSV columns to float: {e}")

        # train/test split (time-aware: first part -> train)
        split_idx = int(len(self.values) * split)
        self.train_data = self.values[:split_idx]
        self.test_data  = self.values[split_idx:]

        if verbose:
            print(f"[DataLoader] Read {len(self.values)} rows -> train {len(self.train_data)} / test {len(self.test_data)}")

        # scaler placeholders
        self.min_val = None
        self.max_val = None

    def minmax_scale_fit(self, data):
        minv = data.min(axis=0)
        maxv = data.max(axis=0)
        self.min_val = minv
        self.max_val = maxv
        return (data - minv) / (maxv - minv + 1e-9)

    def minmax_scale_transform(self, data):
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Call minmax_scale_fit first.")
        return (data - self.min_val) / (self.max_val - self.min_val + 1e-9)

    def create_sequences(self, data, seq_len):
        seqs = []
        # create windows of exact length seq_len (sliding)
        for i in range(len(data) - seq_len + 1):
            seqs.append(data[i:i+seq_len])
        return np.array(seqs)

    def prepare_data(self, seq_len, normalise=True, val_fraction=0.1, verbose=True):
        """
        Returns: X_train, X_val, X_test
        (For autoencoder, y == X)
        val_fraction: fraction of train to use for validation (time-aware: tail of train)
        """
        # scale
        if normalise:
            train_scaled = self.minmax_scale_fit(self.train_data)
            test_scaled  = self.minmax_scale_transform(self.test_data)
        else:
            train_scaled = self.train_data
            test_scaled  = self.test_data

        # create sequences
        x_train_all = self.create_sequences(train_scaled, seq_len)
        x_test      = self.create_sequences(test_scaled, seq_len)

        # create validation as tail slice of train (time-aware)
        if val_fraction and val_fraction > 0:
            n_train = len(x_train_all)
            n_val = max(1, int(n_train * val_fraction))
            # keep last n_val windows as validation
            X_train = x_train_all[:-n_val]
            X_val   = x_train_all[-n_val:]
        else:
            X_train = x_train_all
            X_val = None

        if verbose:
            print(f"[DataLoader] x_train shape: {X_train.shape}, x_val shape: {None if X_val is None else X_val.shape}, x_test shape: {x_test.shape}")

        return X_train, X_val, x_test

    def inverse_transform(self, scaled_array):
        """Inverse minmax transform"""
        return scaled_array * (self.max_val - self.min_val + 1e-9) + self.min_val
