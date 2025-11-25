# hardware_control.py
import logging
import time
import serial
from serial.tools import list_ports
import threading

from config import (GPIO, RPI_GPIO_AVAILABLE, STEPS_PER_UM_Z, STEPS_PER_UM_XY)

# LightControl 保持不變...
class LightControl:
    def __init__(self, bf_pin, fluo_pin, status_callback):
        self.bf_pin = bf_pin; self.fluo_pin = fluo_pin
        self.status_callback = status_callback
        if RPI_GPIO_AVAILABLE:
            try:
                GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.bf_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(self.fluo_pin, GPIO.OUT, initial=GPIO.LOW)
                self.status_callback("Light controller initialized.")
                logging.info(f"Initialized GPIO pins: Bright Field={bf_pin}, Fluorescence={fluo_pin}")
            except Exception as e:
                self.status_callback(f"GPIO Error: {e}"); logging.error(f"Failed to initialize GPIO: {e}")
        else:
            self.status_callback("Light controller disabled (No GPIO).")
    def set_light(self, pin, state):
        if RPI_GPIO_AVAILABLE:
            output_state = GPIO.HIGH if state else GPIO.LOW
            GPIO.output(pin, output_state)
            light_name = "Bright Field" if pin == self.bf_pin else "Fluorescence"
            status = "ON" if state else "OFF"
            self.status_callback(f"{light_name} light turned {status}.")
            logging.info(f"Set GPIO pin {pin} to {status}")
        else:
            light_name = "Bright Field" if pin == self.bf_pin else "Fluorescence"
            status = "ON" if state else "OFF"
            logging.warning(f"Simulating: Turn {light_name} light {status} (GPIO pin {pin}).")
            self.status_callback("Cannot control light (No GPIO).")
    def cleanup(self):
        if RPI_GPIO_AVAILABLE:
            GPIO.cleanup(); logging.info("GPIO cleanup complete.")

class MotorConversion:
    @staticmethod
    def um_to_microsteps_z(um):
        return int(round(um * STEPS_PER_UM_Z))

    @staticmethod
    def um_to_microsteps_xy(um):
        return int(round(um * STEPS_PER_UM_XY))

class MotorControl:
    def __init__(self, status_callback):
        self.ser = None
        self.status_callback = status_callback
        self.port = self._find_arduino_port()
        self.lock = threading.Lock()
        
        if self.port:
            try:
                self.ser = serial.Serial(self.port, 9600, timeout=0.1) 
                time.sleep(2) 
                self.status_callback(f"Connected to Arduino on {self.port}")
                logging.info(f"Successfully connected to Arduino on {self.port}")
            except serial.SerialException as e:
                self.status_callback(f"Error connecting to Arduino: {e}")
                logging.error(f"Failed to connect to Arduino on {self.port}: {e}")
                self.ser = None
        else:
            self.status_callback("Arduino not found.")
            logging.warning("Could not find an Arduino connected.")

    def _find_arduino_port(self):
        ports = list_ports.comports()
        for port in ports:
            if "arduino" in port.description.lower() or "ch340" in port.description.lower() or port.vid == 0x2341:
                return port.device
        return None

    def send_command(self, command, stop_flag=lambda: False, silent=False):
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    final_command = command
                    original_command = command
                    
                    if command.startswith('s'):
                        final_command = command[1:]
                    elif command[0].lower() in ['x', 'y', 'z']:
                        axis = command[0].lower()
                        value_str = command[1:]
                        if value_str not in ['0', 'center']: 
                            try:
                                um_val = float(value_str)
                                if axis == 'z':
                                    microsteps = MotorConversion.um_to_microsteps_z(abs(um_val))
                                elif axis == 'x' or axis == 'y':
                                    microsteps = MotorConversion.um_to_microsteps_xy(abs(um_val))
                                
                                microsteps_with_dir = microsteps if um_val >= 0 else -microsteps
                                final_command = f"{axis}{microsteps_with_dir}" 
                            except ValueError:
                                logging.error(f"Invalid motor val: {value_str}")
                                if not silent: self.status_callback(f"Invalid: {value_str}")
                                return

                    self.ser.reset_input_buffer() 
                    self.ser.write(f"{final_command}\n".encode('utf-8'))
                    
                    if not silent:
                        self.status_callback(f"Sent: {original_command} -> {final_command}. Waiting for ACK...")
                        logging.info(f"Sent to Arduino: {final_command}")
                    
                    if 'center' in final_command or final_command == 'z0':
                         return 

                    response = ""
                    start_time = time.time()
                    
                    while time.time() - start_time < 5.0:
                        if stop_flag():
                            logging.warning(f"Command {final_command} cancelled.")
                            return
                        
                        # 【Crash Fix】 Check if serial is still valid before reading
                        if not self.ser or not self.ser.is_open:
                            return

                        try:
                            line = self.ser.readline()
                            if line:
                                response = line.decode('utf-8').strip()
                                if response:
                                    if not silent:
                                        self.status_callback(f"Arduino ACK: {response}")
                                        logging.info(f"Arduino ACK: {response}")
                                    return
                        except Exception as e:
                            logging.error(f"Read error: {e}")
                            return
                        
                    if not silent:
                        self.status_callback(f"Warning: No ACK for {final_command}")
                    logging.warning(f"No ACK from Arduino for {final_command}")

                except serial.SerialException as e:
                    if not silent: self.status_callback(f"Serial Error: {e}")
                    logging.error(f"Serial Error: {e}")
            else:
                if not silent: self.status_callback("Arduino not connected.")
                logging.warning(f"Arduino not connected.")

    def close(self):
        with self.lock:
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.ser = None # Explicitly set to None
                logging.info("Serial port closed.")