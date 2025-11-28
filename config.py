# config.py
import logging
import os

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- RPi.GPIO Check ---
try:
    import RPi.GPIO as GPIO
    RPI_GPIO_AVAILABLE = True
except (RuntimeError, ImportError):
    GPIO = None 
    RPI_GPIO_AVAILABLE = False
    logging.warning("WARNING: RPi.GPIO library not found. Light control will be disabled.")

# --- Camera Settings ---
RAW_SIZE = (3280, 2464)

FLIP_VERTICAL = False   # 上下翻轉
FLIP_HORIZONTAL = True # 左右翻轉

# --- 預覽和效能選項 ---
RESOLUTION_OPTIONS = {
    "Low (410x308)": (820, 616),
    "Medium (820x616)": (1640, 1232),
    "High (1024x768)": (2048, 1536),
    "Full (1640x1232)": (3280, 2464),
}

FPS_OPTIONS = {
    "15": 15,
    "30": 30,
    "60": 60,
}

DEFAULT_RESOLUTION_KEY = "Medium (820x616)"
DEFAULT_FPS_KEY = "30"

# --- GPIO Pin Definitions ---
BRIGHT_FIELD_PIN = 17
FLUORESCENCE_PIN = 27

# --- Motor/CNC Shield Settings (NEMA 17) ---
MOTOR_STEPS_PER_REV = 200       
MICROSTEP_DIVIDER = 16          
Z_LEAD_SCREW_MM = 2.0           
XY_LEAD_SCREW_MM = 2.0          

STEPS_PER_UM_Z = (MOTOR_STEPS_PER_REV * MICROSTEP_DIVIDER) / Z_LEAD_SCREW_MM / 1000.0
STEPS_PER_UM_XY = (MOTOR_STEPS_PER_REV * MICROSTEP_DIVIDER) / XY_LEAD_SCREW_MM / 1000.0

# --- Autofocus Settings ---
FOCUS_FOLDER = "/home/pi/Desktop/GUI/autofocus_picture" 

FOCUS_STEPS_LIST = [
    (100, 'laplacian', True),
    (30, 'laplacian', True),
    (10, 'laplacian', True),
    (3, 'laplacian', False),
    (1, 'laplacian', False),
]

FOCUS_STEPS_DRIFT = [
    (3, 'laplacian', False),
    (1, 'laplacian', False),
]

# --- Acquisition Protocol Settings ---
CAPTURE_FOLDER = "/home/pi/Desktop/GUI/pifp_data"
PROTOCOL_RANGE_UM = 100        
PROTOCOL_NUM_IMAGES = 25       

# --- Correction Settings (新增) ---
# 請確認您的 .npy 檔案位於此路徑
UNMIX_MATRIX_PATH = "/home/pi/Desktop/GUI/streaming/unmix_tensor.npy"