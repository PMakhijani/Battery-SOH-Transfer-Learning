"""
data_pipeline.py — Preprocessing for LFP and NMC data.

Changes:
  - 4 inputs: [V, I, T, dQ/dV]       (was 2)
  - Resample to 360 points   (was 200)
  - SOH = Q_max(n) / Q_max(1)   (was fixed / 1.1)
  - Smooth before diff (was after)
  - Scaler on train only        (was per cycle)
  - Split by cell           (was random)
"""

import os
import glob
import pickle
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

SEQ_LEN = 360    # 1 hour @ 10s
SIGMA_LFP = 8            # Smooth for LFP
SIGMA_NMC = 3            # Smooth for NMC


# ─────────────────────────────────────────────────────────────────────────────
# Step A: Resample time
# ─────────────────────────────────────────────────────────────────────────────
def resample_to_fixed_length(t, *channels, target_len=SEQ_LEN):
    """
    Interpolate to fixed length.
    Handles variable inputs.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        t = np.linspace(0, 1, max(len(channels[0]), 2))

    t_uniform = np.linspace(t[0], t[-1], target_len)
    out = []
    for ch in channels:
        ch = np.asarray(ch, dtype=float)
        if ch.size < 2:
            out.append(np.zeros(target_len))
            continue
        # This interpolation stuff is kinda confusing but works
        t_ch = np.linspace(t[0], t[-1], ch.size)
        f = interp1d(t_ch, ch, kind='linear',
                     bounds_error=False, fill_value=(ch[0], ch[-1]))
        out.append(f(t_uniform))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step B: ICA (dQ/dV) with Gaussian pre-smoothing
# ─────────────────────────────────────────────────────────────────────────────
def compute_dqdv(V, Q, sigma=SIGMA_LFP):
    """
    Incremental Capacity Analysis: smooth Q on voltage grid, then differentiate.
    Gaussian smoothing BEFORE differentiation (proposal Step B).
    Returns dQ/dV in the same time-order index as input arrays.
    """
    sort_idx = np.argsort(V)
    V_s = V[sort_idx]
    Q_s = Q[sort_idx]

    Q_smooth = gaussian_filter1d(Q_s.astype(float), sigma=sigma)
    dV = np.gradient(V_s)
    dqdv_sorted = np.gradient(Q_smooth) / (dV + 1e-9)

    # Map back to original time order
    inv_idx = np.argsort(sort_idx)
    return dqdv_sorted[inv_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Step D: Scaler (fit on train split only)
# ─────────────────────────────────────────────────────────────────────────────
def fit_scaler(dataset):
    """Per-channel min/max computed exclusively from training data."""
    X_all = np.array([d['X'] for d in dataset], dtype=np.float32)  # (N, 360, 4)
    scaler = {
        'mean': X_all.mean(axis=(0, 1)),   # shape (4,)
        'std': X_all.std(axis=(0, 1)),
    }
    # prevent division by zero for flat channels
    scaler['std'][scaler['std'] < 1e-9] = 1e-9
    return scaler


def apply_scaler(dataset, scaler):
    """Normalize each channel to [0, 1] using training-set statistics."""
    for d in dataset:
        d['X'] = ((d['X'] - scaler['mean']) / scaler['std']).astype(np.float32)
    return dataset


def save_scaler(scaler, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Step E: Cell-level train / val / test split
# ─────────────────────────────────────────────────────────────────────────────
def split_by_cell(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Split by CELL ID — prevents data leakage where cycles from the same
    cell appear in both train and test sets.
    """
    cell_ids = np.array(sorted(set(d['cell_idx'] for d in dataset)))
    rng = np.random.default_rng(seed)
    rng.shuffle(cell_ids)

    n = len(cell_ids)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))

    train_ids = set(cell_ids[:i1])
    val_ids   = set(cell_ids[i1:i2])
    test_ids  = set(cell_ids[i2:])

    train = [d for d in dataset if d['cell_idx'] in train_ids]
    val   = [d for d in dataset if d['cell_idx'] in val_ids]
    test  = [d for d in dataset if d['cell_idx'] in test_ids]

    print(f"Cell split -> train: {len(train_ids)} cells / {len(train)} cycles | "
          f"val: {len(val_ids)} cells / {len(val)} cycles | "
          f"test: {len(test_ids)} cells / {len(test)} cycles")
    return train, val, test


def dataset_to_arrays(dataset):
    """Convert list of cycle dicts to (X, y) numpy arrays."""
    X = np.array([d['X'] for d in dataset], dtype=np.float32)   # (N, 360, 4)
    y = np.array([d['SOH'] for d in dataset], dtype=np.float32)  # (N,)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MIT-Stanford LFP — Source Domain
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_ref(f, entry):
    """Dereference an HDF5 object reference to a flat numpy array."""
    try:
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            if entry.size == 1:
                el = entry.item()
                if isinstance(el, h5py.h5r.Reference):
                    return f[el][()].flatten()
                return np.array(el).flatten()
            return np.array(entry).flatten()
        if isinstance(entry, h5py.h5r.Reference):
            return f[entry][()].flatten()
        return np.array(entry).flatten()
    except Exception:
        return np.array([])


def _extract_channel(f, cell_obj, key, cyc_idx):
    """Safely extract and flatten a channel from the HDF5 cell object."""
    if key not in cell_obj:
        return None
    try:
        return _resolve_ref(f, cell_obj[key][cyc_idx, 0])
    except Exception:
        return None


def _get_Q(f, cell_obj, cyc_idx):
    """
    Return discharge capacity array using key-priority lookup.
    Avoids `arr or arr` ValueError on numpy arrays.
    """
    for key in ('Qd', 'Qc', 'Qdis'):
        arr = _extract_channel(f, cell_obj, key, cyc_idx)
        if arr is not None and arr.size > 0:
            return arr
    return None


def process_mit_lfp(file_path, max_cells=None, verbose=True):
    """
    Load MIT-Stanford LFP HDF5 and build 4-channel cycle tensors.

    Returns: list of dicts with keys:
        cell_idx, cycle_idx, X (360,4), SOH, chemistry, temperature_C
    """
    dataset = []

    with h5py.File(file_path, 'r') as f:
        cycles_col = f['batch']['cycles']
        n_cells = int(cycles_col.shape[0])
        if max_cells:
            n_cells = min(n_cells, max_cells)
        if verbose:
            print(f"MIT-Stanford LFP: {n_cells} cells found in HDF5.")

        for cell_idx in range(n_cells):
            try:
                cell_ref = cycles_col[cell_idx, 0]
                cell_obj = f[cell_ref]

                if not (isinstance(cell_obj, h5py.Group) and 'V' in cell_obj):
                    continue

                n_cycles = int(cell_obj['V'].shape[0])
                q_cycle1 = None
                n_ok = 0

                for cyc_idx in range(n_cycles):
                    try:
                        V = _extract_channel(f, cell_obj, 'V', cyc_idx)
                        Q = _get_Q(f, cell_obj, cyc_idx)
                        I = _extract_channel(f, cell_obj, 'I', cyc_idx)
                        T = _extract_channel(f, cell_obj, 'T', cyc_idx)
                        # 't' in MIT dataset is time in minutes
                        t_raw = _extract_channel(f, cell_obj, 't', cyc_idx)

                        if V is None or Q is None or V.size < 10:
                            continue

                        # SOH = relative to cycle-1 max discharge
                        q_max = float(Q.max())
                        if q_cycle1 is None:
                            q_cycle1 = q_max
                        if q_cycle1 < 1e-6:
                            continue
                        soh = q_max / q_cycle1
                        if soh > 1.1 or soh < 0.5:
                            continue

                        # Graceful fallbacks for missing channels
                        if I is None:
                            I = np.full_like(V, -q_max)
                        if T is None:
                            T = np.full(V.size, 25.0)
                        # Convert minutes → seconds
                        t_sec = (t_raw * 60.0) if t_raw is not None else np.arange(V.size) * 10.0

                        # Resample all channels to 360 time steps
                        V_r, I_r, T_r, Q_r = resample_to_fixed_length(t_sec, V, I, T, Q)

                        # ICA: Gaussian-smoothed dQ/dV (σ=8 for LFP)
                        dqdv = compute_dqdv(V_r, Q_r, sigma=SIGMA_LFP)

                        X = np.stack([V_r, I_r, T_r, dqdv], axis=-1).astype(np.float32)

                        dataset.append({
                            'cell_idx':     cell_idx,
                            'cycle_idx':    cyc_idx,
                            'X':            X,           # (360, 4)
                            'SOH':          np.float32(soh),
                            'chemistry':    'LFP',
                            'temperature_C': float(np.mean(T_r)),
                        })
                        n_ok += 1

                    except Exception:
                        continue

                if verbose and n_ok > 0:
                    print(f"  Cell {cell_idx:2d}: {n_ok} cycles processed.")

            except Exception as e:
                if verbose:
                    print(f"  Cell {cell_idx}: FAILED — {e}")

    print(f"\nMIT-Stanford LFP total: {len(dataset)} cycles.")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# CALCE NMC — Target Domain (CS2_35, CS2_36 xlsx files)
# ─────────────────────────────────────────────────────────────────────────────
# Actual CALCE CS2 sheet: 'Channel_1-008'
# Columns confirmed: Test_Time(s), Cycle_Index, Current(A), Voltage(V),
#                    Charge_Capacity(Ah), Discharge_Capacity(Ah)
# Note: No temperature column → default 25°C

_DATA_SHEET_PREFIX = 'Channel_'   # CALCE data lives in Channel_* sheets

def _load_calce_file(path):
    """Read one CALCE xlsx file; return the data sheet as a DataFrame."""
    xl = pd.ExcelFile(path, engine='openpyxl')
    data_sheets = [s for s in xl.sheet_names if s.startswith(_DATA_SHEET_PREFIX)]
    if not data_sheets:
        return None
    return pd.read_excel(xl, sheet_name=data_sheets[0])


def process_calce_nmc(data_dirs, sigma=SIGMA_NMC, cell_idx_offset=1000, verbose=True):
    """
    Load all CALCE NMC .xlsx files from one or more directories.

    IMPORTANT: Each xlsx file has its own Test_Time(s) counter starting from 0
    and Cycle_Index starting from 1. Files MUST be processed independently and
    stitched with a global cycle counter — do NOT concat all files before groupby.

    Args:
        data_dirs:        str or list of str — paths to CS2_35/, CS2_36/, ...
        sigma:            Gaussian σ for ICA smoothing (NMC default = 3)
        cell_idx_offset:  offset so NMC cell IDs don't clash with LFP IDs

    Returns: list of cycle dicts
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    dataset = []

    for dir_offset, data_dir in enumerate(sorted(data_dirs)):
        dir_name  = os.path.basename(data_dir)
        cell_id   = cell_idx_offset + dir_offset

        files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
        if verbose:
            print(f"CALCE: loading {len(files)} files from {data_dir} ...")

        q_cycle1     = None
        n_ok         = 0
        global_cycle = 0   # monotonic counter across all files for this cell

        for fp in files:
            try:
                df = _load_calce_file(fp)
                if df is None:
                    continue

                # Sort within this file by Cycle_Index then Test_Time
                df = df.sort_values(['Cycle_Index', 'Test_Time(s)']).reset_index(drop=True)

                for cyc_id, grp in df.groupby('Cycle_Index', sort=True):
                    grp = grp.reset_index(drop=True)

                    # CALCE CS2 Arbin: discharge = Current(A) < 0 (Step 7, I≈-1.1A)
                    # Discharge_Capacity(Ah) resets to 0 each cycle within a file
                    dis = grp[grp['Current(A)'] < 0].copy()
                    if len(dis) < 10:
                        continue

                    V = dis['Voltage(V)'].values.astype(float)
                    I = dis['Current(A)'].values.astype(float)
                    Q = dis['Discharge_Capacity(Ah)'].values.astype(float)
                    t = dis['Test_Time(s)'].values.astype(float)
                    T = np.full(len(V), 25.0)   # No temperature channel in CS2

                    q_max = float(Q.max())
                    if q_max < 0.05:
                        continue

                    if q_cycle1 is None:
                        q_cycle1 = q_max

                    soh = q_max / q_cycle1
                    if soh > 1.1 or soh < 0.5:
                        continue

                    global_cycle += 1
                    V_r, I_r, T_r, Q_r = resample_to_fixed_length(t, V, I, T, Q)
                    dqdv = compute_dqdv(V_r, Q_r, sigma=sigma)
                    X = np.stack([V_r, I_r, T_r, dqdv], axis=-1).astype(np.float32)

                    dataset.append({
                        'cell_idx':      cell_id,
                        'cycle_idx':     global_cycle,
                        'X':             X,
                        'SOH':           np.float32(soh),
                        'chemistry':     'NMC',
                        'temperature_C': 25.0,
                    })
                    n_ok += 1

            except Exception as e:
                if verbose:
                    print(f"  Warning: could not process {os.path.basename(fp)} - {e}")
                continue

        if verbose:
            print(f"  {dir_name} (cell {cell_id}): {n_ok} discharge cycles.")

    print(f"\nCALCE NMC total: {len(dataset)} cycles.")
    return dataset

