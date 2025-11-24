# core/model.py
import os
import datetime as dt
import numpy as np
from tensorflow.keras.models import Model as KerasModel, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from core.utils import Timer

class AutoencoderModel:
    def __init__(self):
        self.autoencoder = None
        self.threshold = None
        self.built = False

    def load_model(self, filepath):
        self.autoencoder = load_model(filepath, compile=False)
        self.built = True
        print(f"[Model] Loaded model from {filepath}")

    def build_model(self, configs):
        timer = Timer(); timer.start()
        seq_len = configs['data']['sequence_length']
        n_features = len(configs['data']['columns'])

        enc_layers = configs['model'].get('encoder_layers', [64, 32])
        dec_layers = configs['model'].get('decoder_layers', enc_layers[::-1])
        latent_dim = configs['model'].get('latent_dim', enc_layers[-1] if enc_layers else 32)
        dropout = configs['model'].get('dropout', 0.2)
        loss = configs['model'].get('loss', 'huber')
        optimizer = configs['model'].get('optimizer', 'adam')

        inputs = Input(shape=(seq_len, n_features), name='input_sequence')
        x = inputs

        # Encoder stack
        for i, units in enumerate(enc_layers):
            return_seq = True if i < (len(enc_layers)-1) else False
            x = LSTM(units, return_sequences=return_seq)(x)
            x = BatchNormalization()(x)
            if dropout and dropout > 0:
                x = Dropout(dropout)(x)

        # If final encoder output is 3D -> make 2D via LSTM
        if len(x.shape) == 3:
            x = LSTM(latent_dim, return_sequences=False)(RepeatVector(1)(x))
        encoded = Dense(latent_dim, activation='relu')(x)

        # Decoder
        x = RepeatVector(seq_len)(encoded)
        for units in dec_layers:
            x = LSTM(units, return_sequences=True)(x)
            x = BatchNormalization()(x)
            if dropout and dropout > 0:
                x = Dropout(dropout)(x)

        outputs = TimeDistributed(Dense(n_features))(x)

        self.autoencoder = KerasModel(inputs, outputs, name='lstm_autoencoder')

        if loss == 'huber':
            loss_fn = Huber()
        else:
            loss_fn = loss

        self.autoencoder.compile(optimizer=optimizer, loss=loss_fn)
        self.built = True
        print(f"[Model] Built autoencoder seq_len={seq_len}, n_features={n_features}")
        timer.stop()

    def train(self, x_train, x_val=None, epochs=10, batch_size=32, save_dir='models'):
        if not self.built:
            raise RuntimeError("Model not built.")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss' if x_val is not None else 'loss', patience=5, restore_best_weights=True),
            ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss' if x_val is not None else 'loss')
        ]
        history = self.autoencoder.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, x_val) if x_val is not None else None,
            callbacks=callbacks,
            shuffle=False  # time-series: do not shuffle
        )
        # ModelCheckpoint already saved best; save final as well
        self.autoencoder.save(save_path)
        print(f"[Model] Training finished. Best saved to {save_path}")
        return history

    def reconstruct(self, x):
        if self.autoencoder is None:
            raise RuntimeError("No trained autoencoder.")
        return self.autoencoder.predict(x, verbose=0)

    def reconstruction_error(self, x, reduction='mean'):
        rec = self.reconstruct(x)
        mse_per_timestep = np.mean(np.square(x - rec), axis=2)
        if reduction == 'mean':
            return np.mean(mse_per_timestep, axis=1)
        elif reduction == 'sum':
            return np.sum(mse_per_timestep, axis=1)
        else:
            return mse_per_timestep

    def estimate_threshold(self, x_val, multiplier=3.0, reduction='mean'):
        errs = self.reconstruction_error(x_val, reduction=reduction)
        mu = np.mean(errs); sigma = np.std(errs)
        self.threshold = mu + multiplier * sigma
        print(f"[Model] Threshold set: {self.threshold:.6e} (mean {mu:.6e}, std {sigma:.6e})")
        return self.threshold

    def detect_anomalies(self, x, reduction='mean'):
        if self.threshold is None:
            raise RuntimeError("Threshold not estimated.")
        errs = self.reconstruction_error(x, reduction=reduction)
        return errs > self.threshold

    def save_threshold(self, path):
        if self.threshold is None:
            raise RuntimeError("No threshold to save.")
        with open(path, 'w') as f:
            f.write(str(self.threshold))
        print(f"[Model] Threshold saved to {path}")

    def load_threshold(self, path):
        with open(path, 'r') as f:
            self.threshold = float(f.read().strip())
        print(f"[Model] Threshold loaded from {path}: {self.threshold}")
