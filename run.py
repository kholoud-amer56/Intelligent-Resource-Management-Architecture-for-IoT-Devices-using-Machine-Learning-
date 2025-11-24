# run.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from core.data_processor import DataLoader
from core.model import AutoencoderModel
from core.utils import Timer

plt.style.use('seaborn') if 'seaborn' in plt.style.available else None

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_if_not_exists(arr, path, fmt="%.6e"):
    if os.path.exists(path):
        print(f"[INFO] {path} already exists -> skipping save.")
    else:
        np.savetxt(path, arr.reshape(arr.shape[0], -1), delimiter=",", fmt=fmt)
        print(f"[INFO] Saved {path}.")

def plot_recon_errors(errors, threshold, anomalies_idx, outpath=None):
    plt.figure(figsize=(14,4))
    plt.plot(errors, label='Reconstruction error')
    plt.hlines(threshold, xmin=0, xmax=len(errors), colors='r', label='Threshold')
    plt.scatter(anomalies_idx, errors[anomalies_idx], color='r', label='Anomalies')
    plt.legend()
    plt.xlabel('Window index')
    plt.ylabel('Error (MSE mean)')
    if outpath:
        plt.savefig(outpath)
    plt.show()

def main():
    configs = json.load(open('config.json', 'r'))
    ensure_dir(configs['model']['save_dir'])

    # Load & prepare data
    print("[INFO] Loading data...")
    dl = DataLoader(configs['data']['filename'], configs['data']['train_test_split'], configs['data']['columns'])
    seq_len = configs['data']['sequence_length']

    X_train, X_val, X_test = dl.prepare_data(seq_len, normalise=configs['data']['normalise'], val_fraction=0.1)
    print(f"[INFO] x_train shape: {X_train.shape}, x_val shape: {None if X_val is None else X_val.shape}, x_test shape: {X_test.shape}")

    # Build model
    print("[INFO] Building model...")
    model = AutoencoderModel()
    model.build_model(configs)

    # Train (use X_val for validation)
    print("[INFO] Training model...")
    timer = Timer(); timer.start()
    history = model.train(X_train, x_val=X_val, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'], save_dir=configs['model']['save_dir'])
    timer.stop()

    # Estimate threshold from validation (preferred) otherwise from part of train
    if X_val is not None:
        thr_source = X_val
    else:
        thr_source = X_train[: max(1, int(0.1 * len(X_train)))]

    print("[INFO] Estimating threshold...")
    threshold = model.estimate_threshold(thr_source, multiplier=3.0, reduction='mean')
    model.save_threshold(os.path.join(configs['model']['save_dir'], 'threshold.txt'))

    # Reconstruct test set and compute errors
    print("[INFO] Reconstructing test set...")
    rec = model.reconstruct(X_test)
    errors = np.mean(np.square(X_test - rec), axis=(1,2))  # per-window MSE mean

    # anomaly mask
    anomalies = errors > threshold
    anomalies_idx = np.where(anomalies)[0]
    print(f"[INFO] Test windows: {len(errors)}, anomalies detected: {anomalies.sum()}")

    # Save results if not exist
    ensure_dir('results')
    save_if_not_exists(rec, os.path.join('results', 'reconstructed_test.csv'))
    save_if_not_exists(errors, os.path.join('results', 'reconstruction_error_test.csv'))
    # Also save binary mask
    save_if_not_exists(anomalies.astype(int), os.path.join('results', 'anomaly_mask_test.csv'), fmt="%d")

    # Plot reconstruction errors + anomalies
    plot_recon_errors(errors, threshold, anomalies_idx, outpath=os.path.join('results', 'recon_errors.png'))

    # Summary metrics (reconstruction)
    mae = np.mean(np.abs(X_test.reshape(len(X_test), -1) - rec.reshape(len(rec), -1)))
    mse = np.mean(np.square(X_test.reshape(len(X_test), -1) - rec.reshape(len(rec), -1)))
    rmse = np.sqrt(mse)
    print("--------------------------------------------------")
    print("✅ [RESULTS] Model Reconstruction Metrics (Batch Mode):")
    print(f"   MAE: {mae:.6e}")
    print(f"   MSE: {mse:.6e}")
    print(f"   RMSE: {rmse:.6e}")

if __name__ == '__main__':
    main()
