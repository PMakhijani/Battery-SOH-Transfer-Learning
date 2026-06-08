"""
train_lfp.py — Phase 1: Train on LFP.

Changes:
  - Split by cell 70/15/15 (no overlap)
  - Scaler on train only, save it
  - Right SOH calc
  - 4 inputs (360,4) not (200,2)
  - Callbacks: Early stop, reduce LR, checkpoint
  - Save history
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

# Run from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (
    process_mit_lfp, split_by_cell,
    fit_scaler, apply_scaler, save_scaler, dataset_to_arrays,
)
from src.model import build_base_brain
from src.evaluate import evaluate_model, plot_soh_vs_gt, plot_loss_curves

# Config
MIT_FILE        = 'MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat'
MODEL_SAVE_PATH = 'models/base_brain_lfp.h5'
SCALER_SAVE_PATH= 'models/scaler_lfp.pkl'
EPOCHS          = 200
BATCH_SIZE      = 16


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Load LFP
    print('=' * 60)
    print('PHASE 1: Loading LFP data ...')
    print('=' * 60)
    dataset = process_mit_lfp(MIT_FILE, verbose=True)
    if not dataset:
        print(f"No data, check {MIT_FILE}")
        return

    # Split
    print('\nSplitting by cell ID (70 / 15 / 15) ...')
    train_data, val_data, test_data = split_by_cell(dataset)

    # 3. Scaler — fit on TRAIN only ────────────────────────────────────────────
    print('Fitting normalisation scaler on train set ...')
    scaler = fit_scaler(train_data)
    train_data = apply_scaler(train_data, scaler)
    val_data   = apply_scaler(val_data,   scaler)
    test_data  = apply_scaler(test_data,  scaler)
    save_scaler(scaler, SCALER_SAVE_PATH)
    print(f'Scaler saved -> {SCALER_SAVE_PATH}')

    # 4. Convert to numpy arrays ───────────────────────────────────────────────
    X_train, y_train = dataset_to_arrays(train_data)
    X_val,   y_val   = dataset_to_arrays(val_data)
    X_test,  y_test  = dataset_to_arrays(test_data)
    print(f'\nX_train: {X_train.shape}   y_train: {y_train.shape}')
    print(f'X_val:   {X_val.shape}     y_val:   {y_val.shape}')
    print(f'X_test:  {X_test.shape}    y_test:  {y_test.shape}')

    # 5. Build model ───────────────────────────────────────────────────────────
    model = build_base_brain(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    # 6. Train ─────────────────────────────────────────────────────────────────
    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7,
            min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='val_loss',
            save_best_only=True, verbose=1),
    ]

    print(f'\nTraining on {len(X_train)} cycles ...')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1,
    )

    # 7. Evaluate on held-out LFP test set ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('EVALUATION - Held-out LFP Test Set')
    print('=' * 60)
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = evaluate_model(y_test, y_pred, label='LFP Test')

    # 8. Save history & plots ──────────────────────────────────────────────────
    np.save('results/lfp_train_history.npy', history.history)
    plot_loss_curves(history.history, save_name='plot3_loss_curves_lfp.png')

    # Plot 1: SOH curve for first test cell
    test_cell_ids = sorted(set(d['cell_idx'] for d in test_data))
    if test_cell_ids:
        first_cell = [d for d in test_data if d['cell_idx'] == test_cell_ids[0]]
        fc_X, fc_y = dataset_to_arrays(first_cell)
        fc_pred    = model.predict(fc_X, verbose=0).flatten()
        plot_soh_vs_gt(fc_y, fc_pred,
                       title=f'SOH Prediction - LFP Cell {test_cell_ids[0]} (Test)',
                       save_name='plot1_soh_lfp.png')

    # Plot 7: cell-level heatmap on test data
    from src.evaluate import plot_cell_error_heatmap
    plot_cell_error_heatmap(test_data, model, save_name='plot7_cell_heatmap_lfp.png')

    print(f'\nModel saved -> {MODEL_SAVE_PATH}')
    print('LFP pre-training complete.')
    return history, metrics


if __name__ == '__main__':
    main()
