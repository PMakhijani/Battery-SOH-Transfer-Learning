"""
evaluate.py — All metrics (MAE, RMSE, MAPE, R²) + 7 required plots + ablation study.

Fixes vs calclulate_metrics.py:
  - Adds MAPE (was missing, explicitly required in proposal)
  - Fixes filename typo (calclulate → evaluate)
  - Adds all 7 required plots
  - Adds ablation study helper
"""

import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIGURES_DIR = 'results/figures'


def _ensure():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, label=''):
    """Compute and print RMSE, MAE, MAPE, R². Returns dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100   # %
    mae  = mean_absolute_error(y_true, y_pred) * 100            # %
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100  # %
    r2   = r2_score(y_true, y_pred)

    tag = f'[{label}] ' if label else ''
    print(f'{tag}RMSE: {rmse:.2f}%  |  MAE: {mae:.2f}%  |  MAPE: {mape:.2f}%  |  R²: {r2:.4f}')
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — SOH time-series: predicted vs ground truth (LFP test cell)
# ─────────────────────────────────────────────────────────────────────────────
def plot_soh_vs_gt(y_true, y_pred, title='SOH Prediction vs Ground Truth (LFP)',
                   save_name='plot1_soh_lfp.png'):
    _ensure()
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Actual SOH', color='royalblue', linewidth=1.5)
    plt.plot(y_pred, label='Predicted SOH (Bi-LSTM)', color='tomato',
             linestyle='--', linewidth=1.5)
    plt.title(title, fontsize=14)
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SOH)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Scatter: predicted vs actual (NMC cross-chemistry)
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter(y_true, y_pred, title='Predicted vs Actual SOH (NMC Transfer)',
                 save_name='plot2_scatter_nmc.png'):
    _ensure()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12, color='steelblue')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], 'r--', label='Perfect fit')
    plt.xlabel('Actual SOH')
    plt.ylabel('Predicted SOH')
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Training loss curves (LFP pre-training + optional NMC fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────
def plot_loss_curves(history_lfp, history_nmc=None,
                     save_name='plot3_loss_curves.png'):
    _ensure()
    ncols = 2 if history_nmc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    axes = np.atleast_1d(axes)

    axes[0].plot(history_lfp['loss'],     label='Train', color='royalblue')
    axes[0].plot(history_lfp['val_loss'], label='Val',   color='tomato')
    axes[0].set_title('LFP Pre-training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Huber Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if history_nmc and len(axes) > 1:
        axes[1].plot(history_nmc['loss'],     label='Train', color='royalblue')
        axes[1].plot(history_nmc['val_loss'], label='Val',   color='tomato')
        axes[1].set_title('NMC Fine-tuning Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Huber Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — ICA comparison: LFP vs NMC dQ/dV curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_ica_comparison(lfp_samples, nmc_samples, n_curves=5,
                        save_name='plot4_ica_comparison.png'):
    _ensure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_curves))

    for ax, samples, title in [(ax1, lfp_samples, 'LFP — dQ/dV (ICA)'),
                                (ax2, nmc_samples, 'NMC — dQ/dV (ICA)')]:
        for j, s in enumerate(samples[:n_curves]):
            ax.plot(s['X'][:, 3], color=colors[j], alpha=0.8,
                    label=f"cycle {s['cycle_idx']}")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Resampled Time Step')
        ax.set_ylabel('dQ/dV (a.u.)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Shot-efficiency: RMSE vs number of NMC fine-tuning cycles
# ─────────────────────────────────────────────────────────────────────────────
def plot_shot_efficiency(shot_results, save_name='plot5_shot_efficiency.png'):
    """shot_results: list of (n_cycles, rmse_pct) tuples."""
    _ensure()
    ns    = [r[0] for r in shot_results]
    rmses = [r[1] for r in shot_results]
    plt.figure(figsize=(8, 5))
    plt.plot(ns, rmses, marker='o', color='steelblue', linewidth=2)
    plt.axhline(5.0, color='red', linestyle='--', label='5% RMSE target')
    plt.xlabel('NMC Fine-tuning Cycles')
    plt.ylabel('RMSE (%)')
    plt.title('Shot-Efficiency Curve', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 — Temperature error comparison: 25°C vs 45°C
# ─────────────────────────────────────────────────────────────────────────────
def plot_temp_error_comparison(y_true_25, y_pred_25, y_true_45, y_pred_45,
                               save_name='plot6_temp_error.png'):
    _ensure()
    err_25 = np.abs(np.asarray(y_true_25) - np.asarray(y_pred_25)) * 100
    err_45 = np.abs(np.asarray(y_true_45) - np.asarray(y_pred_45)) * 100

    plt.figure(figsize=(8, 5))
    plt.boxplot([err_25, err_45],
                labels=['25°C (Nominal)', '45°C (Indian Summer)'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='steelblue'))
    plt.ylabel('Absolute SOH Error (%)')
    plt.title('Thermal Robustness: Error at 25°C vs 45°C', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 — Cell-level error heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_cell_error_heatmap(dataset, model, save_name='plot7_cell_heatmap.png'):
    """Per-cell mean absolute error bar chart."""
    _ensure()
    from collections import defaultdict
    cell_errors = defaultdict(list)

    for d in dataset:
        X = d['X'][np.newaxis]
        y_true = float(d['SOH'])
        y_pred = float(model.predict(X, verbose=0)[0][0])
        cell_errors[d['cell_idx']].append(abs(y_true - y_pred) * 100)

    cell_ids   = sorted(cell_errors)
    mean_errs  = [np.mean(cell_errors[c]) for c in cell_ids]

    plt.figure(figsize=(10, max(5, len(cell_ids) * 0.3)))
    plt.barh(range(len(cell_ids)), mean_errs, color='steelblue', alpha=0.75)
    plt.yticks(range(len(cell_ids)), [f'Cell {c}' for c in cell_ids])
    plt.axvline(5.0, color='red', linestyle='--', label='5% target')
    plt.xlabel('Mean Absolute Error (%)')
    plt.title('Cell-Level SOH Prediction Error', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Study — 4 configurations (proposal Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation_study(nmc_dataset, n_ft=50, lfp_model_path='models/base_brain_lfp.h5'):
    """
    Runs 4-config ablation. Requires a trained LFP model at lfp_model_path.
    Returns list of (config_name, rmse_pct) tuples.
    """
    import tensorflow as tf
    from code_stuff.data_pipeline import dataset_to_arrays, fit_scaler, apply_scaler
    from code_stuff.battery_model import build_base_brain

    scaler = fit_scaler(nmc_dataset[:n_ft])
    nmc_scaled = apply_scaler(copy.deepcopy(nmc_dataset), scaler)
    X_ft,   y_ft   = dataset_to_arrays(nmc_scaled[:n_ft])
    X_eval, y_eval = dataset_to_arrays(nmc_scaled[n_ft:])

    if len(X_eval) == 0:
        print('Not enough NMC cycles for ablation eval set (need > n_ft cycles).')
        return []

    def _fit_and_eval(m, label, epochs=50, lr=1e-4):
        m.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss=tf.keras.losses.Huber(1.0), metrics=['mae'])
        m.fit(X_ft, y_ft, epochs=epochs, batch_size=8, validation_split=0.2,
              verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                          restore_best_weights=True)])
        y_pred = m.predict(X_eval, verbose=0).flatten()
        return evaluate_model(y_eval, y_pred, label=label)

    results = []

    # Config 1: Scratch (no LFP pre-training)
    print('\n[Ablation 1/4] Scratch — train on NMC only')
    m1 = build_base_brain()
    r1 = _fit_and_eval(m1, 'Scratch', epochs=100)
    results.append(('Baseline (scratch)', r1['rmse']))

    # Config 2: Transfer, No Freeze
    print('\n[Ablation 2/4] Transfer — all layers trainable')
    m2 = tf.keras.models.load_model(lfp_model_path)
    for lyr in m2.layers: lyr.trainable = True
    r2 = _fit_and_eval(m2, 'Transfer, No Freeze')
    results.append(('Transfer, No Freeze', r2['rmse']))

    # Config 3: Transfer + Freeze  (proposed)
    print('\n[Ablation 3/4] Transfer + Freeze (proposed)')
    m3 = tf.keras.models.load_model(lfp_model_path)
    for lyr in m3.layers:
        lyr.trainable = ('bi_lstm' not in lyr.name)
    r3 = _fit_and_eval(m3, 'Transfer + Freeze')
    results.append(('Transfer + Freeze (proposed)', r3['rmse']))

    # Config 4: Transfer + Freeze + Progressive Unfreeze
    print('\n[Ablation 4/4] Transfer + Freeze + Progressive Unfreeze')
    for lyr in m3.layers:
        if lyr.name == 'bi_lstm_2':
            lyr.trainable = True
    m3.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0),
               loss=tf.keras.losses.Huber(1.0), metrics=['mae'])
    m3.fit(X_ft, y_ft, epochs=30, batch_size=8, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,
                       restore_best_weights=True)])
    y_pred4 = m3.predict(X_eval, verbose=0).flatten()
    r4 = evaluate_model(y_eval, y_pred4, 'Transfer + Freeze + Prog. Unfreeze')
    results.append(('Transfer + Freeze + Prog. Unfreeze', r4['rmse']))

    # Print table
    print('\n' + '=' * 65)
    print(f"{'Configuration':<45} {'RMSE':>8}")
    print('-' * 65)
    for name, rmse in results:
        mark = '  <- TARGET' if rmse < 5.0 else ''
        print(f'{name:<45} {rmse:>6.2f}%{mark}')
    print('=' * 65)

    return results
