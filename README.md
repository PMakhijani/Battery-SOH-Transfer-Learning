# SABARMATI Spark v2.0
## Edge-AI Driven Distributed Battery Management System for Real-Time State Estimation

> **Summer Research Internship Program - SRIP**
> IIT Gandhinagar

**Author:** Piyush Makhijani

---

# Overview

SABARMATI Spark v2.0 is an Edge-AI Driven Distributed Battery Management System (BMS) designed for lithium-ion battery packs.

Unlike conventional centralized Battery Management Systems, this project separates safety-critical hardware from embedded intelligence and AI prediction using a three-tier architecture.

The complete system integrates

- Hardware Analog Protection (BQ76920)
- Embedded Telemetry (STM32F411)
- Edge AI (Jetson Orin Nano)
- Deep Learning based SOH Prediction
- Extended Kalman Filter SOC Estimation
- Real-Time Digital Twin Dashboard
- Hardware-in-the-Loop Fault Injection

The objective is to perform **offline**, **low-latency**, and **fault-tolerant** battery monitoring without relying on cloud infrastructure.

---

# Features

## Tier 1 – Analog Protection Layer

- Texas Instruments BQ76920 Analog Front-End
- Hardware over-voltage / under-voltage protection
- Hardware over-current detection
- MOSFET isolation
- Independent autonomous protection
- Sub-millisecond battery disconnect

---

## Tier 2 – Embedded Telemetry Layer

Running on

- STM32F411CE (ARM Cortex-M4)

Features

- High-speed battery telemetry acquisition
- Cell voltage monitoring
- Pack current monitoring
- Temperature monitoring
- Extended Kalman Filter (EKF) based SOC estimation
- UART packet generation
- UDP telemetry broadcasting

---

## Tier 3 – Edge AI Layer

Running on

- NVIDIA Jetson Orin Nano

Features

- CNN–BiLSTM Battery SOH estimation
- Incremental Capacity Analysis (dQ/dV)
- Transfer Learning from LFP → NMC
- TensorRT optimized inference
- ONNX Runtime deployment
- Remaining Useful Life (RUL) forecasting
- Digital Twin visualization

---

# System Architecture

```
                   Battery Pack
                        │
                        ▼
        ┌─────────────────────────────────┐
        │ Tier 1 : Analog Protection      │
        │ BQ76920 + MOSFET Isolation      │
        └─────────────────────────────────┘
                        │
                    I²C Bus
                        │
                        ▼
        ┌─────────────────────────────────┐
        │ Tier 2 : STM32F411              │
        │ EKF SOC Estimation              │
        │ Telemetry Processing            │
        │ UART → UDP Bridge               │
        └─────────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────┐
        │ Tier 3 : Jetson Orin Nano       │
        │ CNN-BiLSTM SOH Prediction       │
        │ TensorRT Inference              │
        │ Digital Twin Dashboard          │
        └─────────────────────────────────┘
```

---

# Machine Learning Pipeline

The predictive model estimates battery State of Health (SOH) using a transfer learning strategy.

## Phase 1

Pre-train the model using the MIT-Stanford LFP dataset.

Output

```
models/base_brain_lfp.h5
```

---

## Phase 2

Transfer the learned degradation features to CALCE NMC batteries by freezing the Bi-LSTM backbone and fine-tuning the prediction head.

Output

```
models/transferred_nmc.h5
```

---

## Feature Inputs

Each charging cycle is represented using four synchronized channels.

- Voltage (V)
- Current (I)
- Temperature (T)
- Incremental Capacity (dQ/dV)

The Incremental Capacity Analysis (ICA) captures electrochemical aging signatures that are difficult to observe using voltage alone.

---

# Dataset

## MIT-Stanford Dataset

Chemistry

- Lithium Iron Phosphate (LFP)

Used for

- Base model training

Location

```
MIT/
```

---

## CALCE Dataset

Chemistry

- Lithium Nickel Manganese Cobalt Oxide (NMC)

Used for

- Transfer Learning
- Fine-tuning
- Testing

Location

```
Calce Data/
```

---

# Project Structure

```
ATET/
│
├── src/
│   ├── edge_suite.py
│   ├── model.py
│   ├── train_lfp.py
│   ├── data_pipeline.py
│   └── test_edge_suite.py
│
├── code_stuff/
│   ├── battery_model.py
│   ├── data_pipeline.py
│   ├── train_lfp.py
│   ├── transfer_nmc.py
│   ├── evaluate.py
│   └── thermal_augment.py
│
├── models/
│
├── MIT/
│
├── Calce Data/
│
├── results/
│   └── figures/
│
├── notebooks/
│
├── dashboard.py
├── srip_dashboard.py
├── uart_processor.py
├── check_results.py
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>

cd ATET
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Quick Verification

```bash
python run_pipeline.py --test-data
```

---

## Phase 1 – LFP Pre-training

```bash
python run_pipeline.py --phase 1
```

Expected output

```
models/base_brain_lfp.h5
```

---

## Phase 2 – Transfer Learning

```bash
python run_pipeline.py --phase 2
```

Expected output

```
models/transferred_nmc.h5
```

---

## Complete Pipeline

```bash
python run_pipeline.py
```

---

## Evaluate Model

```bash
python check_results.py
```

Outputs

- RMSE
- MAE
- MAPE
- R²
- Thermal robustness evaluation

---

## Launch Dashboard

Standard dashboard

```bash
streamlit run dashboard.py
```

Scuderia Dashboard

```bash
streamlit run srip_dashboard.py
```

---

# Edge Deployment

The Jetson deployment suite includes

- TensorRT optimized inference
- ONNX Runtime execution
- Hybrid EKF + Neural SOC estimation
- Hybrid Remaining Useful Life forecasting
- UDP telemetry listener

Main module

```
src/edge_suite.py
```

---

# UART Telemetry Pipeline

```
STM32
    │
 UART
    │
    ▼
uart_processor.py
    │
 UDP Broadcast
    │
    ▼
Jetson Nano
    │
    ▼
Dashboard
```

---

# Model Design

The deep learning model consists of

- Convolutional feature extraction
- Bidirectional LSTM temporal encoding
- Dense prediction head
- Transfer Learning
- Progressive Layer Unfreezing

Optimizer

```
Adam
clipnorm = 1.0
```

---

# Important Design Choices

| Decision | Purpose |
|------------|----------|
| SOH = Q(n)/Q(1) | Chemistry-independent label |
| 4-channel input | Voltage, Current, Temperature and ICA |
| dQ/dV features | Capture electrochemical degradation |
| Gaussian smoothing | Noise reduction for ICA curves |
| Cell-level data split | Prevents leakage |
| Train-only scaler | Avoids look-ahead bias |
| Layer freezing | Transfers chemistry-independent features |
| Progressive unfreezing | Improves adaptation to new chemistry |

---

# Performance

The architecture targets transfer learning across battery chemistries while maintaining low inference latency.

| Metric | Value |
|---------|--------|
| SOH MAE | 2.34 % |
| RMSE | 2.53 % |
| MAPE | 2.32 % |
| Inference Latency | < 10 ms |
| Dashboard Refresh | 10 ms |

Thermal robustness experiments are also included to evaluate model performance under elevated temperatures.

---

# Generated Outputs

The evaluation pipeline automatically produces

- SOH prediction plots
- Predicted vs Actual scatter plots
- Training loss curves
- Incremental Capacity comparison
- Transfer learning efficiency
- Temperature robustness plots
- Cell-wise error heatmaps

Saved inside

```
results/figures/
```

---

# Hardware Stack

- Texas Instruments BQ76920
- STM32F411CE Black Pill
- NVIDIA Jetson Orin Nano
- IRLFZ44N Power MOSFETs
- 4S Lithium-ion Battery Pack

---

# Technologies Used

- Python
- TensorFlow / Keras
- ONNX Runtime
- TensorRT
- Streamlit
- NumPy
- Pandas
- Plotly
- Scikit-learn
- SciPy

---

# Future Work

- Support for 400V/800V EV battery packs
- Federated Edge Learning
- ISO 26262 functional safety compliance
- Multi-chemistry transfer learning
- Cloud-assisted fleet analytics
- Hardware acceleration using dedicated NPUs

---

# Acknowledgements

This project was developed as part of the **ES665 – Advanced Transport Electrification** course and SRIP research at **IIT Gandhinagar** under the guidance of **Prof. Pallavi Bharadwaj**.

---

# License

This repository is intended for academic and research purposes.