# Master LLM Prompt: Universal Battery Health Diagnostics via Cross-Chemistry Transfer Learning
**Course:** ES 665 – Advanced Transportation Electrification  
**Team:** Rohan Jariwala & Piyush Makhijani | IIT Gandhinagar  
**Document Purpose:** Complete, executable prompt for an LLM / AI coding assistant to implement the full project end-to-end.

---

## CONTEXT & PROBLEM STATEMENT

You are an expert ML engineer and electrochemical systems researcher. Your task is to build a complete, production-ready Python pipeline for **Universal Battery State-of-Health (SOH) estimation** using **Cross-Chemistry Transfer Learning**.

### Background
Indian Electric Vehicles (EVs) face extreme thermal diversity — ambient temperatures range from 4°C (Shimla/Kashmir) to 45°C+ (Rajasthan). Standard Battery Management System (BMS) models trained in controlled 25°C lab environments fail catastrophically under these real-world thermal stresses. Additionally, collecting aging data for new battery chemistries requires 1–2 years of continuous lab testing — an unacceptable bottleneck for a fast-moving EV industry.

### Core Hypothesis
**Degradation physics (ion diffusion dynamics, capacity fade mechanisms) are chemistry-agnostic; only the voltage operating windows differ between chemistries.** This means a model trained on one chemistry (LFP) can be efficiently adapted to another (NMC) with minimal new data.

---

## PRIMARY OBJECTIVES (All must be fulfilled)

### Objective 1 — Base Brain (Pre-training)
Train a Bidirectional LSTM (Bi-LSTM) model — the "Base Brain" — on high-resolution LFP (Lithium Iron Phosphate) battery cycling data from the MIT-Stanford dataset until it achieves RMSE < 5% on LFP SOH estimation.

### Objective 2 — Cross-Chemistry Transfer
Implement an Inductive Transfer Learning protocol that maps learned LFP degradation physics to NMC (Nickel Manganese Cobalt) battery chemistry using the CALCE dataset, requiring fewer than 50 fine-tuning cycles.

### Objective 3 — Thermal Robustness Validation
Validate model robustness under "Indian Summer" high-temperature conditions (45°C–50°C) using appropriate high-temperature datasets or augmented thermal stress scenarios.

### Metric Goal
Achieve **RMSE < 5%** on unseen NMC battery chemistries after transfer learning.

---

## FULL TECHNICAL SPECIFICATION

### PHASE 1: DATA PIPELINE & PREPROCESSING

#### 1.1 Datasets Required
```
Source Domain (D_S):  MIT-Stanford LFP Dataset
  - URL: https://data.matr.io/1/ (Severson et al., Nature Energy 2019)
  - Format: .mat or .pkl files, ~124 cells, up to 1000+ cycles each
  - Chemistry: LFP (LiFePO4), Voltage range: 2.0V – 3.6V

Target Domain (D_T): CALCE NMC Dataset
  - URL: https://calce.umd.edu/battery-data
  - Format: .xlsx or .csv files
  - Chemistry: NMC (LiNiMnCoO2), Voltage range: 2.7V – 4.2V
  - Use: CS2 or CX2 cell series for fine-tuning (< 50 cycles)

High-Temperature Validation:
  - CALCE cells with temperature metadata at 45°C–50°C, OR
  - Augment existing data with thermal drift simulation (see Section 1.3)
```

#### 1.2 Raw Data Ingestion
For each battery cycle, extract the following time-series vectors:
- `V(t)` — Terminal Voltage [V]
- `I(t)` — Current [A] (negative = discharge)
- `T(t)` — Cell Surface Temperature [°C]
- `Q(t)` — Cumulative Charge Capacity [Ah]
- `SOH` — State of Health label = Q_discharge(cycle_n) / Q_discharge(cycle_1)

```python
# Target data structure per cycle:
{
  'cycle_idx': int,
  'V': np.ndarray,       # shape (N_timesteps,)
  'I': np.ndarray,       # shape (N_timesteps,)
  'T': np.ndarray,       # shape (N_timesteps,)
  'Q': np.ndarray,       # shape (N_timesteps,)
  'SOH': float,          # scalar label in [0, 1]
  'chemistry': str,      # 'LFP' or 'NMC'
  'temperature_C': float # nominal test temperature
}
```

#### 1.3 Signal Conditioning Pipeline
Implement the following preprocessing steps **in this exact order**:

**Step A — Resampling (Temporal Alignment)**
- Uniformly interpolate all time-series to a fixed timestep: **Δt = 10 seconds**
- This aligns MIT (Source) and CALCE (Target) datasets which have different logging frequencies
- Use `scipy.interpolate.interp1d` with `kind='linear'`
- Target sequence length: 360 timesteps (representing a 1-hour discharge at 10s intervals)
- Zero-pad shorter sequences; truncate longer ones

**Step B — Incremental Capacity Analysis (ICA / dQ/dV)**
- Compute the Incremental Capacity curve: `dQ/dV = ΔQ / ΔV`
- Apply **Gaussian smoothing** (σ = 5 voltage bins) before differentiation to mitigate sensor quantization noise
- Use `scipy.ndimage.gaussian_filter1d`
- This transforms raw voltage curves into identifiable electrochemical phase transition signatures
- The ICA peaks reveal lithium staging reactions and SEI growth — chemistry-transferable features

**Step C — Feature Vector Construction**
Per cycle, construct a 4-channel input tensor:
```
X = [V(t), I(t), T(t), dQ/dV(t)]  →  shape: (360, 4)
```

**Step D — Min-Max Normalization**
- Normalize each channel to [0, 1] using statistics computed **only from the training split**
- Store scaler parameters to apply identically to validation and target domain data
- This prevents gradient explosion during LSTM training

**Step E — Train/Val/Test Split**
- LFP Source: 70% train, 15% val, 15% test (split by cell ID, not by cycle, to prevent data leakage)
- NMC Target fine-tuning: First N < 50 cycles for fine-tuning, remaining for evaluation

#### 1.4 Thermal Augmentation (for Indian Summer Validation)
If high-temp datasets are unavailable, implement synthetic thermal stress:
```python
def thermal_augment(V, T_nominal, T_target=45.0):
    """
    Simulate voltage sag under high temperature using Arrhenius scaling.
    Empirical: ~3mV/°C voltage reduction at high temperatures.
    """
    delta_T = T_target - T_nominal
    V_augmented = V - (0.003 * delta_T)  # 3mV per degree Celsius
    T_augmented = np.full_like(T_nominal, T_target)
    return V_augmented, T_augmented
```

---

### PHASE 2: MODEL ARCHITECTURE (Bi-LSTM "Base Brain")

#### 2.1 Architecture Specification
```python
Model: BatterySOH_BiLSTM
Input:  (batch_size, 360, 4)   # (B, T, Features)
Output: (batch_size, 1)         # SOH scalar

Layers (in order):
  1. Masking(mask_value=0.0)           # Ignores zero-padded timesteps
  2. Bidirectional(LSTM(64, return_sequences=True))
     - Forward pass:  captures Ohmic resistance rise & transient dynamics
     - Backward pass: captures end-of-discharge "Knee-point" onset
  3. Bidirectional(LSTM(32, return_sequences=False))
  4. Dense(64, activation='relu')
  5. Dropout(0.3)
  6. Dense(32, activation='relu')
  7. Dense(1, activation='sigmoid')    # Output: SOH in [0, 1]

Total trainable parameters: ~150,000
```

#### 2.2 Loss Function — Huber Loss
```python
loss = tf.keras.losses.Huber(delta=1.0)
```
**Rationale:** Replaces standard MSE. Behaves as L2 for small errors (smooth gradient near optimum) and L1 for large errors (robust against sensor outliers and non-Gaussian BMS noise). Critical for real-world battery data which contains current sensor spikes.

#### 2.3 Training Configuration
```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7),
    ModelCheckpoint('base_brain_lfp.h5', save_best_only=True)
]
epochs = 200
batch_size = 32
```

#### 2.4 Convergence Criterion
Training stops when: `val_RMSE < 5%` (i.e., val_RMSE < 0.05 on normalized SOH)

---

### PHASE 3: TRANSFER LEARNING PROTOCOL (Inductive Transfer)

#### 3.1 Three-Stage Transfer Process

**Stage 1 — Source Domain Pre-training (D_S)**
- Train the full Bi-LSTM model on LFP data to convergence (RMSE < 5%)
- Model learns: universal temporal manifold of ion diffusion, capacity fade patterns, charge-transfer kinetics
- Save weights as: `base_brain_lfp_weights.h5`

**Stage 2 — Manifold Alignment ("The Freeze")**
```python
# Freeze all Bi-LSTM layers — lock the learned physics
for layer in model.layers:
    if 'bidirectional' in layer.name or 'lstm' in layer.name:
        layer.trainable = False

# Re-initialize only the final Dense head
model.layers[-1] = Dense(1, activation='sigmoid', name='nmc_head')
model.compile(optimizer=Adam(lr=1e-4), loss=Huber(delta=1.0))
```
**Why this works:** The Bi-LSTM layers encode chemistry-agnostic ion migration dynamics. Only the final mapping layer needs to learn the new NMC voltage window (2.7V–4.2V vs LFP's 2.0V–3.6V).

**Stage 3 — Target Domain Adaptation (D_T) — Few-Shot Fine-Tuning**
```python
# Fine-tune on N < 50 NMC cycles ONLY
nmc_finetune_data = load_nmc_cycles(n_cycles=50)

history = model.fit(
    nmc_finetune_data['X'], 
    nmc_finetune_data['y'],
    epochs=50,
    batch_size=8,           # Small batch for few-shot regime
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=10)]
)
```

#### 3.2 Progressive Unfreezing (Optional Advanced Step)
If RMSE > 5% after Stage 3, implement progressive unfreezing:
```python
# Unfreeze last LSTM layer and fine-tune further
model.layers[-3].trainable = True  # Second Bi-LSTM layer
model.compile(optimizer=Adam(lr=1e-5), loss=Huber(delta=1.0))
model.fit(nmc_finetune_data['X'], nmc_finetune_data['y'], epochs=30)
```

---

### PHASE 4: VALIDATION & EVALUATION

#### 4.1 Metrics to Compute and Report
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100  # as %
    mae  = mean_absolute_error(y_true, y_pred) * 100           # as %
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # as %
    r2   = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2)
    
    print(f"RMSE: {rmse:.2f}%  |  MAE: {mae:.2f}%  |  MAPE: {mape:.2f}%  |  R²: {r2:.4f}")
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}
```

**Target:** RMSE < 5% on unseen NMC chemistry.

#### 4.2 Evaluation Scenarios (All Three Required)
1. **In-domain LFP Test:** Evaluate Base Brain on held-out LFP test cells
2. **Cross-chemistry NMC Transfer:** Evaluate transferred model on NMC cells not seen during fine-tuning
3. **High-Temperature Stress:** Evaluate on 45°C–50°C data (CALCE high-temp or thermally augmented data)

#### 4.3 Ablation Study
Run and report results for these four configurations:
| Experiment | LFP Pre-training | Layer Freezing | NMC Fine-tune Cycles | Expected RMSE |
|---|---|---|---|---|
| Baseline (scratch) | ✗ | ✗ | All NMC | ~12–18% |
| Transfer, No Freeze | ✓ | ✗ | 50 | ~7–10% |
| Transfer + Freeze (proposed) | ✓ | ✓ | 50 | **< 5%** |
| Transfer + Freeze + Prog. Unfreeze | ✓ | Partial | 50 | **< 4%** |

#### 4.4 Visualization Requirements
Generate and save the following plots:
```python
# Plot 1: SOH Prediction vs Ground Truth (LFP)
# Plot 2: SOH Prediction vs Ground Truth (NMC after transfer)
# Plot 3: Training loss curves (LFP pre-training + NMC fine-tuning)
# Plot 4: ICA (dQ/dV) curves comparison — LFP vs NMC
# Plot 5: RMSE vs number of fine-tuning cycles (shot-efficiency curve)
# Plot 6: SOH prediction error distribution at different temperatures (25°C vs 45°C)
# Plot 7: Confusion matrix / error heatmap across battery cells
```

---

### PHASE 5: CODE STRUCTURE & DELIVERABLES

#### 5.1 Required File Structure
```
atet_battery_project/
├── data/
│   ├── raw/
│   │   ├── mit_stanford_lfp/      # Raw .mat/.pkl files
│   │   └── calce_nmc/             # Raw .xlsx/.csv files
│   └── processed/
│       ├── lfp_dataset.pkl        # Preprocessed tensors
│       └── nmc_dataset.pkl
│
├── src/
│   ├── data_pipeline.py           # All preprocessing (Steps A–E)
│   ├── feature_extraction.py      # ICA/DVA dQ/dV computation
│   ├── model.py                   # Bi-LSTM architecture definition
│   ├── train_lfp.py               # Phase 1: LFP pre-training
│   ├── transfer_nmc.py            # Phase 3: Transfer learning
│   ├── evaluate.py                # Phase 4: All metrics & plots
│   └── thermal_augment.py         # Thermal stress simulation
│
├── models/
│   ├── base_brain_lfp.h5          # Trained LFP model
│   └── transferred_nmc.h5         # Fine-tuned NMC model
│
├── notebooks/
│   └── ATET_Full_Pipeline.ipynb   # End-to-end Jupyter notebook
│
├── results/
│   └── figures/                   # All saved plots
│
├── requirements.txt
└── README.md
```

#### 5.2 requirements.txt
```
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
h5py>=3.8.0
openpyxl>=3.1.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

### PHASE 6: REPORTING REQUIREMENTS

The final report / notebook must include:

1. **Introduction & Motivation** — Indian EV thermal challenge, data bottleneck problem
2. **Dataset Description** — MIT-Stanford LFP and CALCE NMC statistics (n_cells, n_cycles, temperature ranges)
3. **Preprocessing Pipeline** — With ICA plots showing electrochemical signatures for both chemistries
4. **Model Architecture** — Bi-LSTM diagram, parameter count, design rationale for Masking + Huber Loss
5. **LFP Training Results** — Loss curves, RMSE on test set, SOH prediction plots
6. **Transfer Learning Results** — Ablation table, cross-chemistry RMSE, shot-efficiency curve
7. **Thermal Robustness** — RMSE comparison at 25°C vs 45°C
8. **Discussion** — Does the chemistry-agnostic degradation hypothesis hold? Where does the model fail?
9. **Future Work** — Edge deployment (TFLite), Hardware-in-the-Loop with ESP32, sub-zero (4°C) extension

---

## CONSTRAINTS & EDGE CASES TO HANDLE

```
1. Variable-length discharge cycles → Use Masking layer + zero-padding to fixed length
2. Missing temperature channel in some CALCE files → Impute with nominal test temperature
3. LFP flat voltage plateau (makes dQ/dV peaks very sharp) → Increase Gaussian σ to 8 for LFP
4. NMC sloped voltage (dQ/dV peaks broader) → Use σ = 3 for NMC
5. Class imbalance (few end-of-life cycles) → Oversample cycles with SOH < 0.8
6. Gradient explosion in deep LSTM → Gradient clipping: clipnorm=1.0 in Adam optimizer
7. Overfitting during few-shot fine-tuning → L2 regularization (λ=1e-4) on the new Dense head
8. Dataset download failures → Implement fallback synthetic data generator for testing
```

---

## SUCCESS CRITERIA CHECKLIST

Before considering the project complete, verify ALL of the following:

- [ ] Data pipeline runs end-to-end without errors on both LFP and NMC datasets
- [ ] Preprocessing produces correctly shaped tensors: `(N_cycles, 360, 4)`
- [ ] ICA (dQ/dV) curves show identifiable electrochemical peaks for both chemistries
- [ ] Base Brain (LFP) achieves **RMSE < 5%** on LFP test set
- [ ] Transfer to NMC using **< 50 cycles** achieves **RMSE < 5%** on unseen NMC cells
- [ ] High-temperature (45°C) validation scenario is evaluated and reported
- [ ] Ablation study comparing 4 configurations is complete
- [ ] All 7 required plots are generated and saved
- [ ] Model weights saved and reloadable: `base_brain_lfp.h5`, `transferred_nmc.h5`
- [ ] Complete Jupyter notebook runs top-to-bottom without intervention
- [ ] README documents how to reproduce all results

---

## EXECUTION INSTRUCTIONS FOR THE LLM

When implementing this project, follow this exact sequence:

1. **Start with `data_pipeline.py`** — Build and test the full preprocessing chain on a small subset (5 cells) before processing the full dataset.

2. **Implement `feature_extraction.py`** — Verify ICA curves visually before proceeding. Wrong dQ/dV implementation is the most common failure point.

3. **Build and test `model.py`** — Instantiate the model, print summary, verify input/output shapes with a dummy batch.

4. **Run `train_lfp.py`** — Monitor training; if val_loss plateaus above 5% RMSE after 50 epochs, reduce learning rate by 10x.

5. **Run `transfer_nmc.py`** — Start with exactly 50 fine-tuning cycles. Report RMSE. Then run the shot-efficiency curve (5, 10, 20, 30, 50 cycles).

6. **Run `evaluate.py`** — Generate all plots and the final metrics table.

7. **Document everything** in `ATET_Full_Pipeline.ipynb` with markdown cells explaining the physics intuition at each step.

**If RMSE target is not met:** First try progressive unfreezing (Section 3.2). If still failing, increase fine-tuning cycles to 75 and document why the 50-cycle constraint could not be met.

---

*Prompt authored from proposal: "Universal Battery Health Diagnostics via Cross-Chemistry Transfer Learning" — ES 665, IIT Gandhinagar*
