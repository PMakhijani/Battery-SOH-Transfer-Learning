"""
edge_suite.py — Full-stack predictive safety suite for Jetson Nano.
Contains Hybrid SOC Estimation, Autoregressive RUL Forecasting, and ONNX Runtime Inference.
"""

import os
import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit

# -------------------------------------------------------------------------
# 1. Hybrid EKF-NN SOC Estimation
# -------------------------------------------------------------------------
class HybridEKFSOC:
    def __init__(self, capacity_ah, dt=10.0, r0_init=0.05):
        """
        EKF + NN Hybrid SOC Estimator.
        capacity_ah: Nominal capacity of the battery in Amp-Hours.
        dt: Sample time in seconds.
        r0_init: Initial internal resistance.
        """
        self.capacity_ah = capacity_ah
        self.dt = dt
        
        # State vector: [SOC]
        self.x = np.array([[1.0]])  # Start at 100% SOC
        self.P = np.array([[1e-4]]) # Initial covariance
        
        # Process and Measurement noise covariances
        self.Q = np.array([[1e-6]]) # High confidence in coulomb counting
        self.R = np.array([[1e-2]]) # Variance in voltage measurement
        
        self.r0 = r0_init
        
        # A simple NN-like mapping to adjust R0 based on Temperature and Current
        # In a full deployment, this would be a loaded tf.keras model or similar.
        # Here we simulate the NN's output using an empirical Arrhenius-based proxy.
        self._nn_weights = {'T_base': 25.0, 'alpha': 0.001}

    def _ocv_curve(self, soc):
        """Standard NMC Open Circuit Voltage (OCV) polynomial approximation."""
        # Simple polynomial fit for OCV = f(SOC)
        return 3.0 + 1.2 * soc

    def _docv_dsoc(self, soc):
        """Derivative of OCV with respect to SOC."""
        return 1.2

    def _nn_update_resistance(self, I, T):
        """Neural Network proxy to update internal resistance dynamically."""
        # Resistance increases at lower temperatures and higher C-rates
        delta_t = self._nn_weights['T_base'] - T
        r_dynamic = self.r0 + self._nn_weights['alpha'] * max(0, delta_t) + 0.0001 * abs(I)
        return r_dynamic

    def step(self, V_measured, I, T):
        """
        Execute one step of the Hybrid EKF.
        I: Current (A), negative for discharge.
        V_measured: Terminal voltage (V).
        T: Temperature (C).
        """
        # 1. NN corrects physical parameters
        r_dynamic = self._nn_update_resistance(I, T)
        
        # 2. EKF Predict Step (Coulomb Counting)
        # SOC_k = SOC_k-1 + I_k * dt / (3600 * Q)
        u = I * self.dt / (3600.0 * self.capacity_ah)
        x_pred = self.x + u
        P_pred = self.P + self.Q
        
        # 3. EKF Update Step (OCV Feedback)
        soc_pred = float(x_pred[0, 0])
        ocv_pred = self._ocv_curve(soc_pred)
        
        # Predicted terminal voltage: V = OCV + I*R
        v_pred = ocv_pred + I * r_dynamic
        
        # Jacobian of measurement model H = dV/dSOC = dOCV/dSOC
        H = np.array([[self._docv_dsoc(soc_pred)]])
        
        # Kalman Gain: K = P_pred * H^T * (H * P_pred * H^T + R)^-1
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # State Update
        y_residual = V_measured - v_pred
        self.x = x_pred + K * y_residual
        
        # Covariance Update
        I_mat = np.eye(1)
        self.P = (I_mat - K @ H) @ P_pred
        
        # Constrain SOC to [0, 1]
        self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 1.0)
        
        return self.x[0, 0]

# -------------------------------------------------------------------------
# 2. Hybrid RUL Forecasting
# -------------------------------------------------------------------------
class HybridRULForecaster:
    def __init__(self, eol_threshold=0.8):
        """
        Hybrid Exponential + Autoregressive Remaining Useful Life (RUL) Forecaster.
        eol_threshold: End of Life SOH threshold (e.g., 0.8 for 80%).
        """
        self.eol_threshold = eol_threshold
        self.soh_history = []

    def _exp_decay(self, cycle, a, b, c):
        """Exponential decay curve: a * exp(b * cycle) + c"""
        return a * np.exp(b * cycle) + c

    def update_and_predict(self, current_cycle, current_soh):
        """
        Adds current SOH to history, fits the global exponential trend,
        and uses autoregressive correction for local prediction to find EOL.
        """
        self.soh_history.append((current_cycle, current_soh))
        
        if len(self.soh_history) < 10:
            # Not enough data to fit curve reliably
            return None
            
        cycles = np.array([x[0] for x in self.soh_history])
        sohs = np.array([x[1] for x in self.soh_history])
        
        # Fit global exponential trend
        try:
            # P0: initial guesses
            p0 = [1.0, -0.001, 0.0]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(self._exp_decay, cycles, sohs, p0=p0, maxfev=2000)
        except RuntimeError:
            # Fallback if fit fails
            popt = [1.0, -0.0001, 0.0]
            
        a, b, c = popt
        
        # Autoregressive correction (residual tracking)
        # AR(1) error: e_t = alpha * e_{t-1}
        residuals = sohs - self._exp_decay(cycles, a, b, c)
        if len(residuals) > 1:
            alpha = np.sum(residuals[1:] * residuals[:-1]) / (np.sum(residuals[:-1]**2) + 1e-9)
        else:
            alpha = 0.0
            
        current_residual = residuals[-1]
        
        # Forecast forward
        future_cycle = current_cycle
        projected_soh = current_soh
        
        max_projection = 5000 # Prevent infinite loops
        while projected_soh > self.eol_threshold and (future_cycle - current_cycle) < max_projection:
            future_cycle += 1
            trend_val = self._exp_decay(future_cycle, a, b, c)
            # Apply AR decay to residual
            current_residual *= alpha
            projected_soh = trend_val + current_residual
            
        rul = future_cycle - current_cycle
        return rul

# -------------------------------------------------------------------------
# 3. ONNX / TensorRT Real-Time Inference
# -------------------------------------------------------------------------
class EdgeInferenceEngine:
    def __init__(self, model_path_h5, onnx_path="models/transferred_nmc.onnx"):
        """
        Optimized inference wrapper for Jetson Nano using ONNX Runtime / TensorRT.
        Loads H5 model, converts to ONNX if needed, and sets up session.
        """
        self.onnx_path = onnx_path
        
        # Try to import onnxruntime, fallback to TensorFlow if unavailable
        try:
            import onnxruntime as ort
            self.use_ort = True
            
            # Check if ONNX model exists, otherwise convert
            if not os.path.exists(self.onnx_path):
                print(f"ONNX model {self.onnx_path} not found. Attempting conversion...")
                self._convert_to_onnx(model_path_h5, self.onnx_path)
                
            # Setup ORT session with TensorRT execution provider for Jetson Nano
            providers = [
                ('TensorrtExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True
                }),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            
        except ImportError:
            print("onnxruntime not found. Falling back to TensorFlow optimized inference.")
            self.use_ort = False
            self.model = tf.keras.models.load_model(model_path_h5, compile=False)
            
            # Use tf.function for speed graph execution
            @tf.function(jit_compile=True)
            def fast_predict(x):
                return self.model(x, training=False)
            self.fast_predict = fast_predict

    def _convert_to_onnx(self, h5_path, onnx_path):
        try:
            import tf2onnx
            model = tf.keras.models.load_model(h5_path, compile=False)
            spec = (tf.TensorSpec((None, 360, 4), tf.float32, name="input"),)
            tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
            print("Converted H5 to ONNX successfully.")
        except Exception as e:
            print(f"Failed to convert to ONNX: {e}")

    def fast_dqdv(self, V, Q, sigma=3):
        """
        Highly optimized NumPy-based dQ/dV feature extraction.
        Targeting < 5ms execution.
        """
        from scipy.ndimage import gaussian_filter1d
        
        sort_idx = np.argsort(V)
        V_s = V[sort_idx]
        Q_s = Q[sort_idx]
        
        # Pre-smoothing
        Q_smooth = gaussian_filter1d(Q_s.astype(float), sigma=sigma)
        
        dV = np.gradient(V_s)
        dqdv_sorted = np.gradient(Q_smooth) / (dV + 1e-9)
        
        inv_idx = np.argsort(sort_idx)
        return dqdv_sorted[inv_idx]

    def predict(self, V_seq, I_seq, T_seq, Q_seq):
        """
        Executes a full inference pass (preprocessing + inference).
        Expects sequences of shape (360,).
        Target total latency: < 50ms.
        """
        # 1. Fast Preprocessing
        dqdv = self.fast_dqdv(V_seq, Q_seq, sigma=3)
        X = np.stack([V_seq, I_seq, T_seq, dqdv], axis=-1).astype(np.float32)
        X = np.expand_dims(X, axis=0) # Add batch dim: (1, 360, 4)
        
        # 3. Inference
        if self.use_ort:
            outputs = self.session.run(None, {self.input_name: X})
            return float(outputs[0][0][0])
        else:
            return float(self.fast_predict(tf.constant(X))[0][0])

if __name__ == "__main__":
    print("Edge Suite modules loaded successfully.")
