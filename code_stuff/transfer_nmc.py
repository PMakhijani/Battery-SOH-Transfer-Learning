"""
transfer_nmc.py — Phase 3: Cross-chemistry transfer to NMC (CALCE dataset).

Implements the 3-stage inductive transfer protocol from proposal Section 3:
  Stage 1 — LFP Base Brain already trained (run train_lfp.py first)
  Stage 2 — Freeze all Bi-LSTM layers ("Manifold Alignment")
  Stage 3 — Fine-tune only the Dense head on ≤ 50 NMC cycles

Also runs:
  - Shot-efficiency curve (5, 10, 20, 30, 50 cycles)
  - Thermal robustness validation via thermal_augment.py
  - Saves transferred model to models/transferred_nmc.h5
"""

import os
import sys
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_stuff.data_pipeline import (
    process_calce_nmc, fit_scaler, apply_scaler,
    save_scaler, dataset_to_arrays,
)
from code_stuff.evaluate import (
    evaluate_model, plot_soh_vs_gt, plot_scatter,
    plot_loss_curves, plot_ica_comparison,
    plot_shot_efficiency, plot_temp_error_comparison,
    run_ablation_study,
)
from code_stuff.thermal_augment import augment_dataset_thermal

# ── Config ────────────────────────────────────────────────────────────────────
CALCE_DIRS       = ['Calce Data/CS2_35', 'Calce Data/CS2_36']
LFP_MODEL_PATH   = 'models/base_brain_lfp.h5'
NMC_MODEL_PATH   = 'models/transferred_nmc.h5'
NMC_SCALER_PATH  = 'models/scaler_nmc.pkl'
N_FINETUNE       = 50
SHOT_CONFIGS     = [5, 10, 20, 30, 50]


def freeze_bilstm(model, lr=1e-4):
    """Stage 2: Lock Bi-LSTM layers; only Dense head remains trainable."""
    for lyr in model.layers:
        lyr.trainable = ('bi_lstm_1' not in lyr.name and 'conv1d' not in lyr.name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=losses.MeanSquaredError(),
        metrics=['mae'],
    )
    frozen = [l.name for l in model.layers if not l.trainable]
    print(f'Frozen layers: {frozen}')
    return model


def progressive_unfreeze(model, X_ft, y_ft, lr=1e-5):
    """Stage 3b: Unfreeze bi_lstm_2 and fine-tune at 10× lower LR."""
    print('Progressive unfreeze: enabling bi_lstm_1 ...')
    for lyr in model.layers:
        if lyr.name == 'bi_lstm_1':
            lyr.trainable = True
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=losses.MeanSquaredError(),
        metrics=['mae'],
    )
    model.fit(X_ft, y_ft, epochs=30, batch_size=8,
              validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)
    return model


def fine_tune(model, X_ft, y_ft, epochs=50):
    """Stage 3: Few-shot fine-tuning with small batch size."""
    history = model.fit(
        X_ft, y_ft,
        epochs=epochs, batch_size=8,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )
    return history


def shot_efficiency_curve(nmc_dataset, scaler, lfp_model_path):
    """Run fine-tuning at different N values; return (n, rmse) pairs."""
    nmc_scaled_all = apply_scaler(copy.deepcopy(nmc_dataset), scaler)
    results = []

    for n in SHOT_CONFIGS:
        if n >= len(nmc_scaled_all):
            continue
        X_ft,   y_ft   = dataset_to_arrays(nmc_scaled_all[:n])
        X_eval, y_eval = dataset_to_arrays(nmc_scaled_all[n:])
        if len(X_eval) == 0:
            continue

        m = tf.keras.models.load_model(lfp_model_path, compile=False)
        m = freeze_bilstm(m)
        m.fit(X_ft, y_ft, epochs=50, batch_size=8, validation_split=0.2,
              verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

        y_pred = m.predict(X_eval, verbose=0).flatten()
        metrics = evaluate_model(y_eval, y_pred, label=f'Shot n={n}')
        results.append((n, metrics['rmse']))

    return results


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # 1. Load CALCE NMC data ───────────────────────────────────────────────────
    print('=' * 60)
    print('PHASE 3: Cross-Chemistry Transfer - LFP -> NMC')
    print('=' * 60)
    nmc_dataset = process_calce_nmc(CALCE_DIRS, verbose=True)
    if not nmc_dataset:
        raise RuntimeError('No NMC data loaded. Check CALCE_DIRS.')

    # 2. NMC scaler (fit on fine-tuning portion only) ──────────────────────────
    finetune_raw = nmc_dataset[:N_FINETUNE]
    eval_raw     = nmc_dataset[N_FINETUNE:]

    scaler_nmc = fit_scaler(finetune_raw)
    finetune_data = apply_scaler(copy.deepcopy(finetune_raw), scaler_nmc)
    eval_data     = apply_scaler(copy.deepcopy(eval_raw),     scaler_nmc)
    save_scaler(scaler_nmc, NMC_SCALER_PATH)

    X_ft,   y_ft   = dataset_to_arrays(finetune_data)
    X_eval, y_eval = dataset_to_arrays(eval_data)
    print(f'Fine-tune set: {X_ft.shape}  |  Eval set: {X_eval.shape}')

    # 3. Load LFP Base Brain ───────────────────────────────────────────────────
    print(f'\nLoading LFP Base Brain from {LFP_MODEL_PATH} ...')
    model = tf.keras.models.load_model(LFP_MODEL_PATH, compile=False)

    # 4. Stage 2: Freeze Bi-LSTM ──────────────────────────────────────────────
    model = freeze_bilstm(model)

    # 5. Stage 3: Fine-tune ────────────────────────────────────────────────────
    print(f'\nFine-tuning on {N_FINETUNE} NMC cycles ...')
    history_nmc = fine_tune(model, X_ft, y_ft)

    # 6. Evaluate ──────────────────────────────────────────────────────────────
    if len(X_eval) > 0:
        y_pred = model.predict(X_eval, verbose=0).flatten()
        metrics = evaluate_model(y_eval, y_pred,
                                 label=f'NMC Transfer ({N_FINETUNE} cycles)')

        # Progressive unfreeze if RMSE > 5%
        if metrics['rmse'] > 3.0:
            print(f'\nRMSE {metrics["rmse"]:.2f}% > 3% - progressive unfreezing ...')
            model = progressive_unfreeze(model, X_ft, y_ft)
            y_pred = model.predict(X_eval, verbose=0).flatten()
            metrics = evaluate_model(y_eval, y_pred,
                                     label='NMC + Progressive Unfreeze')

        # Plot 2: NMC scatter
        y_pred_ft = model.predict(X_ft, verbose=0).flatten()
        y_true_all = np.concatenate([y_ft, y_eval])
        y_pred_all = np.concatenate([y_pred_ft, y_pred])

        plot_scatter(y_true_all, y_pred_all,
                     title=f'NMC Transfer SOH — Predicted vs Actual (Train + Eval)',
                     save_name='plot2_scatter_nmc.png')

        # Plot 1 (NMC): time-series SOH curve (Train + Eval)
        plot_soh_vs_gt(y_true_all, y_pred_all,
                       title='SOH Prediction vs Ground Truth (NMC Train + Eval)',
                       save_name='plot1_soh_nmc.png')

    # 7. Plot 3: loss curves (NMC fine-tuning)
    try:
        lfp_hist = np.load('results/lfp_train_history.npy', allow_pickle=True).item()
        plot_loss_curves(lfp_hist, history_nmc.history,
                         save_name='plot3_loss_curves_combined.png')
    except FileNotFoundError:
        plot_loss_curves(history_nmc.history, save_name='plot3_loss_nmc.png')

    # 8. Plot 4: ICA comparison LFP vs NMC (raw unscaled for visual clarity)
    try:
        from code_stuff.data_pipeline import process_mit_lfp
        lfp_samples = process_mit_lfp(
            'MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat',
            max_cells=1, verbose=False)
        plot_ica_comparison(lfp_samples, nmc_dataset[:5],
                            save_name='plot4_ica_comparison.png')
    except Exception as e:
        print(f'Plot 4 skipped: {e}')

    # 9. Plot 5: Shot-efficiency curve
    print('\nRunning shot-efficiency curve ...')
    shot_results = shot_efficiency_curve(nmc_dataset, scaler_nmc, LFP_MODEL_PATH)
    if shot_results:
        plot_shot_efficiency(shot_results, save_name='plot5_shot_efficiency.png')

    # 10. Plot 6: Thermal robustness (25°C vs 45°C augmented)
    if len(eval_data) > 0:
        eval_45 = augment_dataset_thermal(copy.deepcopy(eval_raw), T_target=45.0)
        eval_45 = apply_scaler(eval_45, scaler_nmc)
        X_45, y_45 = dataset_to_arrays(eval_45)
        y_pred_45 = model.predict(X_45, verbose=0).flatten()
        evaluate_model(y_45, y_pred_45, label='NMC @ 45°C (Thermal)')
        plot_temp_error_comparison(
            y_eval, y_pred, y_45, y_pred_45,
            save_name='plot6_temp_error.png')

    # 11. Ablation study
    print('\nRunning ablation study ...')
    run_ablation_study(nmc_dataset, n_ft=N_FINETUNE,
                       lfp_model_path=LFP_MODEL_PATH)

    # 12. Save final model
    model.save(NMC_MODEL_PATH)
    print(f'\nTransferred model saved -> {NMC_MODEL_PATH}')
    print('Transfer learning complete.')


if __name__ == '__main__':
    main()
