# Battery Health Stuff with ML
**ES 665 – Advanced Transport Electrification | IIT G**  
Rohan & Piyush

---

## What we wanna do
Figure out battery health (SOH) for LFP and NMC batteries using some ML model trained on one and transferred to another.  
Goal: Get error less than 5% on new battery type.

---

## Setup
```bash
pip install -r requirements.txt
```

### Data we used
| Dataset | Where | Why |
|---|---|---|
| MIT LFP | `MIT/2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Training the base model |
| CALCE NMC CS2_35 | `Calce Data/CS2_35/*.xlsx` | Fine-tuning and testing |
| CALCE NMC CS2_36 | `Calce Data/CS2_36/*.xlsx` | More fine-tuning |

---

## How to run

### Quick test (just to check if it works, takes like 2 min)
```bash
python run_pipeline.py --test-data
```

### Phase 1: Train on LFP
```bash
python run_pipeline.py --phase 1
# Gets: models/base_brain_lfp.h5  |  Error should be < 5%
```

### Phase 2: Transfer to NMC
```bash
python run_pipeline.py --phase 2
# Gets: models/transferred_nmc.h5  |  Error < 5%
```

### Run everything
```bash
python run_pipeline.py
```
```

---

## Project Structure
```
ATET/
├── src/
│   ├── data_pipeline.py     # 4-channel preprocessing (V, I, T, dQ/dV)
│   ├── model.py             # 2-layer Bi-LSTM with sigmoid output
│   ├── train_lfp.py         # Phase 1: LFP pre-training
│   ├── transfer_nmc.py      # Phase 2: Transfer learning to NMC
│   ├── evaluate.py          # Metrics (MAE/RMSE/MAPE/R²) + 7 plots
│   └── thermal_augment.py   # Arrhenius thermal stress simulation
├── models/                  # Saved .h5 model weights + scalers
├── results/figures/         # 7 generated plots (PNG)
├── MIT/                     # MIT-Stanford LFP .mat file
├── Calce Data/              # CALCE NMC .xlsx files
├── run_pipeline.py          # End-to-end runner
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **SOH = Q(n) / Q(1)** | Relative label — chemistry-agnostic, no hardcoded nominal capacity |
| **4-channel input [V, I, T, dQ/dV]** | dQ/dV encodes electrochemical phase transitions; T enables thermal generalisation |
| **Gaussian σ=8 (LFP), σ=3 (NMC)** | LFP's flat voltage plateau creates very sharp ICA peaks — heavier smoothing needed |
| **Cell-level train/val/test split** | Prevents data leakage where cycles from same cell appear in both train and test |
| **Scaler fit on train only** | Avoids look-ahead bias from test statistics |
| **Layer freezing during transfer** | Bi-LSTM encodes chemistry-agnostic dynamics; only Dense head maps to new voltage window |
| **clipnorm=1.0 in Adam** | Prevents gradient explosion in deep stacked LSTM |

---

## Expected Outputs

| Config | RMSE |
|---|---|
| Baseline from scratch (NMC only) | ~12–18% |
| Transfer, no freeze | ~7–10% |
| **Transfer + freeze (proposed)** | **< 5%** |
| Transfer + progressive unfreeze | **< 4%** |

**Plots generated in `results/figures/`:**
1. `plot1_soh_lfp.png` — SOH time-series: LFP test cell
2. `plot2_scatter_nmc.png` — Predicted vs actual scatter (NMC)
3. `plot3_loss_curves_combined.png` — Training loss: LFP + NMC
4. `plot4_ica_comparison.png` — dQ/dV signatures: LFP vs NMC
5. `plot5_shot_efficiency.png` — RMSE vs fine-tuning cycles
6. `plot6_temp_error.png` — Error at 25°C vs 45°C (Indian Summer)
7. `plot7_cell_heatmap_lfp.png` — Per-cell MAE heatmap
