import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import socket
import json
import threading
import random
import math
from datetime import datetime
import os
import streamlit.components.v1 as components

# ==============================================================================
# EDGE-TO-FOG UDP LISTENER 
# ==============================================================================
@st.cache_resource
def start_udp_listener():
    latest_data = {"v_cells": [0.0, 0.0, 0.0, 0.0], "current": 0.0, "temp": 25.0, "connected": False}
    def listen():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5005)) 
        sock.settimeout(1.0)
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                payload = json.loads(data.decode('utf-8'))
                latest_data.update(payload)
                latest_data["connected"] = True
            except socket.timeout:
                latest_data["connected"] = False
            except Exception:
                pass
    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    return latest_data

LIVE_HARDWARE_DATA = start_udp_listener()

# ==============================================================================
# CONFIGURATION & SIMULATION SETUP
# ==============================================================================
st.set_page_config(page_title="Scuderia BMS Edge-AI", layout="wide", initial_sidebar_state="expanded")

# Polished CSS: Ultra-Dark Unified Carbon Fiber, Zero Streamlit Defaults
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;700&display=swap');
    
    /* 1. UNIFIED ULTRA-DARK CARBON FIBER BACKGROUND */
    /* Targeting the Main Body, the Sidebar, AND the Top Header to make them all identical */
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
        background: 
            linear-gradient(27deg, #080808 5px, transparent 5px) 0 5px,
            linear-gradient(207deg, #080808 5px, transparent 5px) 10px 0px,
            linear-gradient(27deg, #111111 5px, transparent 5px) 0px 10px,
            linear-gradient(207deg, #111111 5px, transparent 5px) 10px 5px,
            linear-gradient(90deg, #0a0a0a 10px, transparent 10px),
            linear-gradient(#0c0c0c 25%, #050505 25%, #050505 50%, transparent 50%, transparent 75%, #141414 75%, #141414 100%) !important;
        background-color: #030303 !important;
        background-size: 20px 20px !important;
    }
    
    /* Hide the default header background color to let the carbon bleed through */
    header { background-color: transparent !important; }
    
    /* 2. SIDEBAR STYLING */
    [data-testid="stSidebar"] {
        border-right: 2px solid #ff2800;
    }
    .st-emotion-cache-16txtl3 { padding: 2rem 1rem; }
    
    /* 3. CUSTOM STATUS BOXES */
    .status-box-online {
        background-color: rgba(0, 255, 102, 0.05);
        border: 1px solid #00ff66;
        color: #00ff66;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Rajdhani', sans-serif;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 0 8px rgba(0,255,102,0.1);
    }

    /* 4. TYPOGRAPHY & CARDS */
    .ferrari-title { font-family: 'Orbitron', sans-serif; color: #ff2800; text-align: center; text-transform: uppercase; letter-spacing: 2px; margin-top: 10px; }
    .ferrari-subtitle { font-family: 'Rajdhani', sans-serif; color: #f0f2f6; text-align: center; font-size: 1.2rem; margin-bottom: 20px; }
    
    .terminal-box { background-color: rgba(3, 3, 3, 0.90); color: #ff2800; font-family: 'Courier New', monospace; padding: 15px; border-radius: 5px; height: 240px; overflow-y: scroll; font-size: 0.85rem; border: 1px solid #ff2800; box-shadow: 0px 0px 15px rgba(255, 40, 0, 0.2); }
    
    .card-container-weak { border: 2px solid #ff2800; background-color: rgba(15, 3, 3, 0.90); border-radius: 8px; padding: 15px; margin-bottom: 10px; box-shadow: 0px 0px 20px rgba(255, 40, 0, 0.4); }
    .card-container-normal { border: 1px solid #222; background-color: rgba(5, 5, 5, 0.90); border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #ff2800; }
    
    .card-title { font-family: 'Orbitron', sans-serif; color: #ffffff !important; font-size: 1.0rem !important; margin-bottom: 8px; text-align: center;}
    .card-text { font-family: 'Rajdhani', sans-serif; color: #cccccc !important; font-size: 1.1rem !important; margin: 4px 0 !important; text-align: center;}
    </style>
""", unsafe_allow_html=True)

if 'log_history' not in st.session_state:
    st.session_state.log_history = [
        f"[{datetime.now().strftime('%H:%M:%S')}] TIER 3: Jetson Orin Nano Edge Core Booting...",
        f"[{datetime.now().strftime('%H:%M:%S')}] TIER 2: STM32 Cortex-M4 Telemetry Bridge Active.",
        f"[{datetime.now().strftime('%H:%M:%S')}] TIER 1: BQ76920 AFE I2C Handshake Confirmed."
    ]
if 'step' not in st.session_state: st.session_state.step = 0
if 'current_temp' not in st.session_state: st.session_state.current_temp = 25.0
if 'fault_active' not in st.session_state: st.session_state.fault_active = False

def log_event(message, level="info"):
    time_str = datetime.now().strftime('%H:%M:%S')
    color = "#ff2800" if level == "alert" else "#f39c12" if level == "warning" else "#ffffff"
    prefix = "🚨" if level == "alert" else "⚠️" if level == "warning" else ""
    st.session_state.log_history.append(f"<span style='color:{color};'>[{time_str}] {prefix} {message}</span>")

def generate_wltp_step(step, mode):
    if mode == "Scuderia Track Mode (Physics Demo)":
        base_voltage, ambient_temp = 3.85, 25.0
        live_current = random.uniform(8.0, 12.0) if random.random() > 0.8 else random.uniform(1.5, 3.5)
        live_current += random.gauss(0, 0.1)
        
        if st.session_state.fault_active:
            live_current = random.uniform(25.0, 35.0) 
            heating_factor = 2.5 
            voltage_sag, cell_3_damage = live_current * 0.05, 0.6 
        else:
            heating_factor = (live_current ** 2) * 0.005  
            voltage_sag, cell_3_damage = live_current * 0.015, 0.0

        cooling_factor = (st.session_state.current_temp - ambient_temp) * 0.05
        st.session_state.current_temp = st.session_state.current_temp + heating_factor - cooling_factor
        
        v_cells = [
            round(base_voltage - voltage_sag + random.gauss(0, 0.005), 3),
            round((base_voltage - 0.02) - voltage_sag + random.gauss(0, 0.005), 3),
            round((base_voltage + 0.01) - voltage_sag - cell_3_damage + random.gauss(0, 0.005), 3), 
            round(base_voltage - voltage_sag + random.gauss(0, 0.005), 3)
        ]
        base_soh = [98.2 + random.gauss(0, 0.1), 98.0, 97.5 - (cell_3_damage * 20), 98.1] 
        return v_cells, live_current, st.session_state.current_temp, base_soh
    else:
        return [3.7, 3.7, 3.7, 3.7], 0.0, 25.0, [92.0, 92.0, 92.0, 92.0]

def create_needle(value, min_val=0, max_val=100, color="#ff2800"):
    normalized = max(0, min(1, (value - min_val) / (max_val - min_val)))
    theta = (1 - normalized) * math.pi
    x_center, y_center, radius = 0.5, 0.22, 0.45
    x_end = x_center + radius * math.cos(theta)
    y_end = y_center + radius * math.sin(theta)
    x_base1, y_base1 = x_center - 0.015 * math.sin(theta), y_center + 0.015 * math.cos(theta)
    x_base2, y_base2 = x_center + 0.015 * math.sin(theta), y_center - 0.015 * math.cos(theta)
    path = f'M {x_base1} {y_base1} L {x_base2} {y_base2} L {x_end} {y_end} Z'
    return [
        dict(type='path', path=path, fillcolor=color, line=dict(color=color), xref='paper', yref='paper'),
        dict(type='circle', xref='paper', yref='paper', x0=0.48, y0=0.18, x1=0.52, y1=0.26, fillcolor='#0a0a0a', line=dict(color=color, width=2))
    ]

# --- FLATTENED BATTERY SVG FUNCTION ---
def get_battery_svg(voltage, min_v=2.5, max_v=4.2):
    percentage = max(0, min(100, ((voltage - min_v) / (max_v - min_v)) * 100))
    color = "#00ff66" if percentage > 50 else "#f39c12" if percentage > 20 else "#ff2800"
    return f'<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;"><svg width="60" height="30" viewBox="0 0 60 30" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="50" height="26" rx="4" ry="4" fill="none" stroke="#ffffff" stroke-width="2"/><rect x="52" y="10" width="4" height="10" rx="2" ry="2" fill="#ffffff"/><rect x="4" y="4" width="{46 * (percentage / 100)}" height="22" rx="2" ry="2" fill="{color}"/></svg></div>'

# ==============================================================================
# SIDEBAR / COMMAND CENTER CONTROLS
# ==============================================================================
st.sidebar.title("🏎️ Scuderia Command Center")
st.sidebar.markdown("---")

chemistry_mode = st.sidebar.selectbox("Active Architecture Mode", ["Scuderia Track Mode (Physics Demo)", "NMC (Virtual HIL Mode)"])

st.sidebar.markdown("### Distributed Hardware Status")
st.sidebar.markdown("<div class='status-box-online'>🧠 TIER 3: Jetson Orin Nano [ONLINE]</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='status-box-online'>🔌 TIER 2: STM32 Cortex-M4 [ONLINE]</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='status-box-online'>🔋 TIER 1: BQ76920 AFE [ONLINE]</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Panelist Demo Controls")
if st.sidebar.button("🚨 Simulate Thermal Fault", type="primary"):
    st.session_state.fault_active = not st.session_state.fault_active
    if st.session_state.fault_active:
        st.toast('🚨 CATASTROPHIC FAULT DETECTED!', icon='🚨')
        log_event("MANUAL OVERRIDE: Injecting Cell 3 Short Circuit.", "alert")
    else:
        st.toast('✅ FAULT CLEARED', icon='🔄')
        log_event("MANUAL OVERRIDE: Clearing fault states. Rebooting...", "info")
        st.session_state.current_temp = 25.0 

st.session_state.step += 1
v_cells, current, temp, soh_cells = generate_wltp_step(st.session_state.step, chemistry_mode)

if st.session_state.step % 10 == 0 and not st.session_state.fault_active:
    log_event(f"BQ76920 I2C Handshake verified. Pack voltage nominal at {sum(v_cells):.2f}V.")

if abs(v_cells[2] - v_cells[0]) > 0.4 and "Asymmetric" not in st.session_state.log_history[-1]:
    log_event(f"CRITICAL: Asymmetric degradation variance isolated in Cell 3! Delta > 400mV", "alert")

# ==============================================================================
# BRANDING HEADER (FERRARI & LAB LOGOS)
# ==============================================================================
h1, h2, h3 = st.columns([1, 4, 1])
with h1:
    if os.path.exists("ferrari_logo.png"): st.image("ferrari_logo.png", use_container_width=True)
with h2:
    st.markdown("<h1 class='ferrari-title'>EDGE-AI BMS ARCHITECTURE</h1>", unsafe_allow_html=True)
    st.markdown("<div class='ferrari-subtitle'>Distributed Telemetry via STM32F411 & Predictive Analytics via Jetson Orin Nano</div>", unsafe_allow_html=True)
with h3:
    if os.path.exists("lab_logo.png"): st.image("lab_logo.png", use_container_width=True)

st.markdown("---")

if st.session_state.fault_active:
    st.error("### ⚠️ CATASTROPHIC HARDWARE FAULT DETECTED ⚠️\n* **Trigger Cause:** Cell 3 Thermal Runaway & High-Current Short Circuit.\n* **Hardware Action:** BQ76920 Analog Protection ACTIVE.", icon="🚨")
    st.markdown("---")

# ==============================================================================
# TIER 1: GLOBAL PACK METRICS (MECHANICAL SPEEDOMETER GAUGES)
# ==============================================================================
mean_soc = max(0.0, min(100.0, 85.4 - (st.session_state.step * 0.01)))
mean_soh = np.mean(soh_cells)

g1, g2, g3 = st.columns([1.5, 1.5, 1])
with g1:
    fig_soc = go.Figure(go.Indicator(
        mode = "gauge+number", value = mean_soc, title = {'text': "PACK SOC (EKF)", 'font': {'size': 20, 'color': '#ffffff', 'family': 'Orbitron'}},
        number = {'font': {'color': '#ffffff', 'size': 45, 'family': 'Orbitron'}, 'suffix': "%"},
        gauge = {'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'family': 'Orbitron'}}, 'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "rgba(10, 10, 10, 0.7)", 'steps': [{'range': [0, 20], 'color': "#220000"}, {'range': [20, 80], 'color': "#111111"}, {'range': [80, 100], 'color': "#222222"}]}
    ))
    fig_soc.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', shapes=create_needle(mean_soc, color="#ff2800"))
    st.plotly_chart(fig_soc, use_container_width=True)

with g2:
    fig_soh = go.Figure(go.Indicator(
        mode = "gauge+number", value = mean_soh, title = {'text': "AI SOH PREDICTION", 'font': {'size': 20, 'color': '#ffffff', 'family': 'Orbitron'}},
        number = {'font': {'color': '#ffffff', 'size': 45, 'family': 'Orbitron'}, 'suffix': "%"},
        gauge = {'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'family': 'Orbitron'}}, 'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "rgba(10, 10, 10, 0.7)", 'steps': [{'range': [0, 50], 'color': "#330000"}, {'range': [50, 80], 'color': "#1a1a1a"}, {'range': [80, 100], 'color': "#052205"}]}
    ))
    needle_color = "#00ff66" if mean_soh > 80 else "#ff2800"
    fig_soh.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', shapes=create_needle(mean_soh, color=needle_color))
    st.plotly_chart(fig_soh, use_container_width=True)

with g3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(label="Pack Temperature", value=f"{temp:.1f} °C", delta=f"{temp - 25.0:.1f} °C Drift" if temp > 26.0 else "Nominal", delta_color="inverse")
    st.metric(label="Telemetry Current", value=f"{current:.2f} A", delta=f"{(current*sum(v_cells)):.1f} W Output")

st.markdown("---")

# ==============================================================================
# LIVE 3D DIGITAL TWIN VIEWER
# ==============================================================================
st.markdown("<h3 style='font-family: Orbitron; color: white;'>🏎️ LIVE 3D SYSTEM ARCHITECTURE</h3>", unsafe_allow_html=True)

# You will replace this URL with your own Spline 3D export link later
spline_url = "https://my.spline.design/untitled-SMTJvox1pY6jnhsOHq34S1Qk/" 

components.html(
    f'''
    <div style="border: 1px solid #ff2800; border-radius: 8px; overflow: hidden; box-shadow: 0px 0px 15px rgba(255, 40, 0, 0.4); background-color: #050505;">
        <iframe src="{spline_url}" frameborder="0" width="100%" height="450px"></iframe>
    </div>
    ''',
    height=455
)
st.markdown("---")


# ==============================================================================
# TIER 2: CELL-WISE TELEMETRY DEEP DIVE (BQ76920 Front-End)
# ==============================================================================
st.markdown("<h3 style='font-family: Orbitron; color: white;'>🔋 ANALOG FRONT-END (BQ76920) TELEMETRY</h3>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
cells_data = zip([c1, c2, c3, c4], range(4), v_cells, soh_cells)

for col, idx, v, soh in cells_data:
    with col:
        is_weak = (v < 3.2)
        card_style = "card-container-weak" if is_weak else "card-container-normal"
        battery_graphic = get_battery_svg(v)
        
        html_content = (
            f'<div class="{card_style}">'
            f'<div class="card-title">CELL {idx+1} DIAGNOSTIC</div>'
            f'{battery_graphic}'
            f'<div class="card-text"><b>VOLTAGE:</b> <span style="color:#ff2800;">{v:.4f} V</span></div>'
            f'<div class="card-text"><b>AI SOH:</b> {soh:.2f}%</div>'
            f'</div>'
        )
        st.markdown(html_content, unsafe_allow_html=True)

st.markdown("---")

# ==============================================================================
# TIER 3: JETSON PHYSICS ENGINE & SAFETY LOGS
# ==============================================================================
st.markdown("<h3 style='font-family: Orbitron; color: white;'>🧠 JETSON ORIN NANO EDGE-AI CORE</h3>", unsafe_allow_html=True)
p_chart, p_logs = st.columns([2, 1])

with p_chart:
    volts_domain = np.linspace(3.1, 4.1, 100) 
    peak_shift = -0.015 
    if st.session_state.fault_active: peak_shift -= 0.15
        
    dqdv_signal = np.exp(-((volts_domain - (3.82 + peak_shift)) / 0.05)**2) * 12.0
    dqdv_signal += np.random.normal(0, 0.15, 100) 
    
    peak_idx = np.argmax(dqdv_signal)
    peak_v = volts_domain[peak_idx]
    peak_y = dqdv_signal[peak_idx]

    fig_dqdv = go.Figure()
    fig_dqdv.add_trace(go.Scatter(x=volts_domain, y=dqdv_signal, mode='lines', name='Live AI dQ/dV', line=dict(color='#ff2800', width=3)))
    fig_dqdv.add_trace(go.Scatter(x=[peak_v], y=[peak_y], mode='markers', name='Peak Tracker', marker=dict(color='#ffffff', size=12, symbol='cross')))
    
    fig_dqdv.update_layout(margin=dict(l=20, r=20, t=10, b=20), height=240, xaxis_title="Voltage (V)", yaxis_title="Incremental Capacity dQ/dV", font=dict(family="Rajdhani", color="#ffffff"), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dqdv, use_container_width=True)

with p_logs:
    if len(st.session_state.log_history) > 30: st.session_state.log_history = st.session_state.log_history[-20:]
    log_html = "<div class='terminal-box'>" + "".join([f"<div>{entry}</div>" for entry in reversed(st.session_state.log_history)]) + "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

time.sleep(1.0)
st.rerun()