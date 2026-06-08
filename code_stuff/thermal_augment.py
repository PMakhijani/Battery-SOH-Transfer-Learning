"""
thermal_augment.py — Synthetic thermal stress simulation for Indian Summer validation.

Physics: ~3 mV/°C voltage sag under elevated temperature (Arrhenius model).
Implemented per proposal Section 1.4.
"""

import copy
import numpy as np


def thermal_augment(sample, T_target=45.0):
    """
    Augment a single cycle sample to simulate high-temperature conditions.

    The 4 channels in sample['X'] are [V, I, T, dQ/dV] (indices 0-3).
    Only V (index 0) and T (index 2) are modified.

    Args:
        sample:   cycle dict with 'X' of shape (360, 4)
        T_target: target temperature in °C (default 45.0 = Indian Summer)

    Returns:
        new sample dict (deep copy) with augmented V and T channels
    """
    aug = copy.deepcopy(sample)
    T_nominal = float(aug['X'][:, 2].mean())
    delta_T = T_target - T_nominal

    if abs(delta_T) < 0.5:      # already close to target — skip
        return aug

    # Voltage sag: 3 mV per °C elevation
    aug['X'][:, 0] -= 0.003 * delta_T   # channel 0: V(t)
    aug['X'][:, 2] = float(T_target)    # channel 2: T(t)
    aug['temperature_C'] = float(T_target)
    return aug


def augment_dataset_thermal(dataset, T_target=45.0):
    """
    Return a new list of thermally augmented copies for every sample.

    Usage:
        dataset_45 = augment_dataset_thermal(test_dataset, T_target=45.0)
    """
    return [thermal_augment(d, T_target=T_target) for d in dataset]
