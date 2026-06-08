"""
run_pipeline.py — This runs the whole thing. Run from the ATET folder.

Usage:
    python run_pipeline.py                    # All phases
    python run_pipeline.py --phase 1          # Just LFP training
    python run_pipeline.py --phase 2          # Just NMC transfer (need phase 1 first)
    python run_pipeline.py --test-data        # Quick check on few cells

Steps:
    Phase 1: src/train_lfp.py   → models/base_brain_lfp.h5
    Phase 2: src/transfer_nmc.py → models/transferred_nmc.h5
"""

import argparse
import sys


def run_phase1():
    print("Starting phase 1, this might take a while...")
    from code_stuff.train_lfp import main as train
    train()
    print("Phase 1 done! Model saved.")


def run_phase2():
    print("Now doing phase 2...")
    from code_stuff.transfer_nmc import main as transfer
    transfer()
    print("Phase 2 finished.")


def run_test():
    """Quick test on 5 LFP and 1 NMC to check shapes."""
    print('\n=== QUICK TEST ===')
    from code_stuff.data_pipeline import (
        process_mit_lfp, process_calce_nmc,
        split_by_cell, fit_scaler, apply_scaler, dataset_to_arrays,
    )
    from code_stuff.battery_model import build_base_brain

    print('\n[1] Loading some LFP cells …')
    lfp_data = process_mit_lfp(
        'MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat',
        max_cells=5, verbose=True)
    if len(lfp_data) == 0:
        print("No LFP data, wtf?")
        return
    if lfp_data[0]['X'].shape != (360, 4):
        print(f"Shape wrong: {lfp_data[0]['X'].shape}")
        return
    soh_vals = [d["SOH"] for d in lfp_data]
    if not all(0.5 <= float(s) <= 1.1 for s in soh_vals):
        print(f"SOH weird: {soh_vals}")
        return
    print(f'  [OK] LFP: {len(lfp_data)} cycles, shape {lfp_data[0]["X"].shape}, '
          f'SOH [{min(soh_vals):.3f}, {max(soh_vals):.3f}]')

    print('\n[2] Loading NMC data (CS2_35) ...')
    nmc_data = process_calce_nmc(['Calce Data/CS2_35'], verbose=True)
    if len(nmc_data) == 0:
        print("No NMC data!")
        return
    if nmc_data[0]['X'].shape != (360, 4):
        print(f"NMC shape bad: {nmc_data[0]['X'].shape}")
        return
    print(f'  [OK] NMC: {len(nmc_data)} cycles, shape {nmc_data[0]["X"].shape}')

    print('\n[3] Scaling and splitting ...')
    train_stuff, val_stuff, test_stuff = split_by_cell(lfp_data)
    scaler_thing = fit_scaler(train_stuff)
    train_stuff = apply_scaler(train_stuff, scaler_thing)
    X_train, y_train = dataset_to_arrays(train_stuff)
    print(f'  [OK] X: {X_train.shape}  y: {y_train.shape}  y_range: [{y_train.min():.3f}, {y_train.max():.3f}]')

    print('\n[4] Testing model ...')
    import numpy as np
    model = build_base_brain()
    preds = model.predict(X_train[:4], verbose=0)
    if preds.shape != (4, 1):
        print(f"Pred shape wrong: {preds.shape}")
        return
    if preds.min() < 0.0 or preds.max() > 1.0:
        print(f"Preds out of range: {preds.flatten()}")
        return
    print(f'  [OK] preds shape: {preds.shape}  vals: {preds.flatten().round(3)}')

    print('\n=== TESTS PASSED ===\n')


def main():
    parser = argparse.ArgumentParser(description='Battery SOH Pipeline')
    parser.add_argument('--phase', type=int, choices=[1, 2],
                        help='Run phase 1 or 2')
    parser.add_argument('--test-data', action='store_true',
                        help='Quick test')
    args = parser.parse_args()

    if args.test_data:
        run_test()
    elif args.phase == 1:
        run_phase1()
    elif args.phase == 2:
        run_phase2()
    else:
        # Full run
        run_phase1()
        run_phase2()


if __name__ == '__main__':
    main()
