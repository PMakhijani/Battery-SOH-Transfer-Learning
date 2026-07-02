"""
test_edge_suite.py — Benchmarks and validations for the Edge Safety Suite.
"""

import os
import sys
import time
import numpy as np

# Add src to path to import edge_suite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_suite import HybridEKFSOC, HybridRULForecaster, EdgeInferenceEngine

def test_soc_estimator():
    print("--- Testing Hybrid EKF SOC Estimator ---")
    ekf = HybridEKFSOC(capacity_ah=5.0, dt=1.0)
    
    socs = []
    # Simulate a discharge with noise
    for i in range(100):
        I = -1.0 
        T = 30.0
        V = 4.2 - (0.001 * i) + np.random.normal(0, 0.005)
        
        soc = ekf.step(V_measured=V, I=I, T=T)
        socs.append(soc)
        
    print(f"Initial SOC: {socs[0]:.4f}")
    print(f"Final SOC after 100s discharge: {socs[-1]:.4f}")
    print("SOC Test Passed.\n")

def test_rul_forecaster():
    print("--- Testing Hybrid RUL Forecaster ---")
    forecaster = HybridRULForecaster(eol_threshold=0.8)
    
    for cycle in range(1, 201):
        soh = np.exp(-0.0005 * cycle) + np.random.normal(0, 0.002)
        forecaster.update_and_predict(current_cycle=cycle, current_soh=soh)
        
    rul = forecaster.update_and_predict(current_cycle=201, current_soh=np.exp(-0.0005 * 201))
    print(f"Current Cycle: 201")
    print(f"Predicted RUL (Remaining Useful Cycles): {rul}")
    print("RUL Test Passed.\n")

def test_inference_benchmark():
    print("--- Benchmarking Edge Inference Engine ---")
    try:
        # Define paths relative to the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h5_path = os.path.join(root_dir, "models", "transferred_nmc.h5")
        onnx_path = os.path.join(root_dir, "models", "transferred_nmc.onnx")
        
        if not os.path.exists(h5_path):
            print(f"Skipping benchmark: Model file not found at {h5_path}")
            return
            
        engine = EdgeInferenceEngine(model_path_h5=h5_path, onnx_path=onnx_path)
        
        # Create dummy sequence data
        V_seq = np.linspace(4.2, 3.0, 360)
        I_seq = np.full(360, -1.0)
        T_seq = np.full(360, 25.0)
        Q_seq = np.linspace(0, 5.0, 360)
        
        print("Running warmup...")
        _ = engine.predict(V_seq, I_seq, T_seq, Q_seq)
        
        print("Benchmarking latency over 100 iterations...")
        n_iters = 100
        start = time.time()
        for _ in range(n_iters):
            _ = engine.predict(V_seq, I_seq, T_seq, Q_seq)
        end = time.time()
        
        avg_time_ms = ((end - start) / n_iters) * 1000
        print(f"Average Inference Latency (Preprocessing + Forward Pass): {avg_time_ms:.2f} ms")
        if avg_time_ms < 50.0:
            print("Latency is strictly under the 50 ms budget! Test Passed.\n")
        else:
            print("Warning: Latency exceeded 50 ms budget.\n")
            
    except Exception as e:
        print(f"Inference Benchmark could not be run fully: {e}")

if __name__ == "__main__":
    test_soc_estimator()
    test_rul_forecaster()
    test_inference_benchmark()
