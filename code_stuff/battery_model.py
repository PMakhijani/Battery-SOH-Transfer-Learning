"""
model.py — Bi-LSTM model thing.

Changes from old version:
  - Two Bi-LSTM layers (64 -> 32 units) not one
  - Added Conv1D + MaxPooling for features
  - Linear output so SOH can be >1
  - MSE loss for better RMSE
  - Adam with clipnorm to stop gradients from exploding
  - Early stopping and stuff in train_lfp.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses


def build_base_brain(input_shape=(360, 4), learning_rate=1e-3):
    """
    BatterySOH_BiLSTM — like in the paper.

    Input:  (batch, 360, 4)  - [V(t), I(t), T(t), dQ/dV(t)]
    Output: (batch, 1)       - SOH value

    Layers:
        Conv1D(64) -> Conv1D(128) -> MaxPooling1D(2) -> BiLSTM(128, seq=True) -> BiLSTM(64, seq=False)
        -> Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dense(1, linear)
    """
    inp = layers.Input(shape=input_shape, name='input')

    # Conv stuff for local features
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv1d_1')(inp)
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', name='conv1d_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool_1')(x)

    # First Bi-LSTM
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True), name='bi_lstm_1')(x)

    # Second Bi-LSTM
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False), name='bi_lstm_2')(x)

    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)

    # Linear for SOH, can be over 1
    out = layers.Dense(1, activation='linear', name='soh_head')(x)

    model = models.Model(inp, out, name='BatterySOH_CNN_BiLSTM')
    # idk why clipnorm helps but it does
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=losses.MeanSquaredError(),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
    )
    return model


if __name__ == '__main__':
    import numpy as np
    m = build_base_brain()
    m.summary()
    dummy = np.random.rand(4, 360, 4).astype('float32')
    out = m.predict(dummy, verbose=0)
    if out.shape != (4, 1):
        print(f"Shape wrong: {out.shape}")
    else:
        print(f"OK: {dummy.shape} -> {out.shape}  vals: {out.flatten()}")
        # this print is just for debugging
