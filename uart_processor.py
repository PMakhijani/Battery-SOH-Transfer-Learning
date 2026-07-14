import serial
import time
import numpy as np
import socket
import json

class UartBmsReceiver:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        
        # Last known good data (fallback if a packet is corrupted)
        self.last_v = [3.2, 3.2, 3.2, 3.2]
        self.last_i = 0.0
        self.last_t = 25.0

    def connect(self):
        """Attempts to open the serial port to the STM32."""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[SUCCESS] Connected to STM32 on {self.port} at {self.baudrate} baud.")
            time.sleep(2) # Give the microcontroller a moment to reset upon connection
            return True
        except serial.SerialException as e:
            print(f"[ERROR] Could not open serial port {self.port}: {e}")
            return False

    def read_latest_packet(self):
        """
        Reads the incoming serial buffer, extracts the latest complete packet, 
        and parses it into floats.
        Expected format: "START,V1,V2,V3,V4,I,T,END\n"
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            return self.last_v, self.last_i, self.last_t

        try:
            # Read whatever is sitting in the buffer
            if self.serial_conn.in_waiting > 0:
                raw_line = self.serial_conn.readline().decode('utf-8').strip()
                
                # Verify packet integrity (Starts with START, ends with END, has exactly 8 chunks)
                packet = raw_line.split(',')
                if len(packet) == 8 and packet[0] == "START" and packet[7] == "END":
                    
                    # Parse the payload
                    v_cells = [float(packet[1]), float(packet[2]), float(packet[3]), float(packet[4])]
                    current = float(packet[5])
                    temp = float(packet[6])
                    
                    # Update fallbacks
                    self.last_v = v_cells
                    self.last_i = current
                    self.last_t = temp
                    
                    return v_cells, current, temp
                else:
                    print(f"[WARNING] Dropped corrupted packet: {raw_line}")
                    
        except Exception as e:
            print(f"[WARNING] Serial read error: {e}")
            
        # If no new data or packet was bad, return the last known good state
        return self.last_v, self.last_i, self.last_t

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("[INFO] Serial connection closed.")



# --- UDP Broadcaster Block ---
if __name__ == "__main__":
    import socket
    import json
    
    # Network Setup: 192.168.55.100 is the default Windows IP when connected to Jetson via USB-C
    UDP_IP = "192.168.55.100" 
    UDP_PORT = 5005
    
    # Create the UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Hardware Setup: Change port if needed (Jetson is usually /dev/ttyUSB0 or /dev/ttyACM0)
    receiver = UartBmsReceiver(port='/dev/ttyUSB0') 
    
    if receiver.connect():
        print(f"Streaming BMS telemetry to {UDP_IP}:{UDP_PORT}... (Press Ctrl+C to stop)")
        try:
            while True:
                # 1. Grab the latest hardware numbers
                v, i, t = receiver.read_latest_packet()
                
                # 2. Package it into a lightweight JSON dictionary
                payload = json.dumps({
                    "v_cells": v, 
                    "current": i, 
                    "temp": t
                })
                
                # 3. Fire it over the network bridge
                sock.sendto(payload.encode('utf-8'), (UDP_IP, UDP_PORT))
                
                # 4. Print locally just so you can see it working on the Jetson monitor
                print(f"SENT -> V: {v} | I: {i}A | T: {t}°C")
                
                time.sleep(0.05) # 50ms loop
                
        except KeyboardInterrupt:
            receiver.close()
            sock.close()
            print("\n[INFO] UDP Stream stopped.")