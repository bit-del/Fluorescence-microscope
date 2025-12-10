# workers.py
import logging
import time
import os
import glob
import cv2
import numpy as np
import json 
import queue
from threading import Lock
import threading 
import math
import shutil 
import csv 

from PySide6.QtCore import (QObject, QSize, Qt, Signal, Slot, QCoreApplication, 
                              QTimer, QEventLoop, QMetaObject, Q_ARG)
from PySide6.QtGui import QPixmap, QImage 
from datetime import datetime

import config
from image_processing import (get_raw_channels, apply_correction, 
                              convert_to_qimage, calculate_gain_maps)

# 【修改】 引入 compute_score_tenengrad
from autofocus import (FocusAlgorithm, compute_score_variance, compute_score_fluo, compute_score_brenner, compute_score_tenengrad) 
from hardware_control import MotorConversion 

class CameraWorker(QObject):
    status_updated = Signal(str)
    controls_locked = Signal(str)
    af_status_finished = Signal() 
    
    def __init__(self, picam2, streaming_output, motor_control):
        super().__init__()
        self.picam2 = picam2
        self.streaming_output = streaming_output
        self.motor_control = motor_control
        
        # --- 影像校正變數 ---
        self.gain_maps = None             
        self.flat_field_channels = None   
        self.bg_frame_raw = None          
        self.bg_subtract_enabled = False  
        
        # 【Unmix Matrix 變數】
        self.raw_unmix_tensor = None      
        self.current_unmix_tensor = None  
        self._load_raw_unmix_tensor()
        
        # 【燈光狀態變數】
        self.is_fluo_mode = False

        self.ev_comp = 1.0
        self.mode = "live_auto"
        self._is_running = True
        
        # --- 相機與串流設定 ---
        self.target_fps = int(config.DEFAULT_FPS_KEY)
        default_w = config.RESOLUTION_OPTIONS[config.DEFAULT_RESOLUTION_KEY][0] // 2
        default_h = config.RESOLUTION_OPTIONS[config.DEFAULT_RESOLUTION_KEY][1] // 2
        self.stream_size = (default_w, default_h)
        self.video_size = (default_w, default_h)
        
        self.video_fps = 10.0
        
        # --- 觸發標記 ---
        self._awb_triggered = False
        self._capture_triggered = False
        self._start_recording_triggered = False
        self._stop_recording_triggered = False
        self.video_writer = None
        
        self._controls_lock = Lock()
        self._pending_controls = None
        
        self.frame_timer = None 
        self.target_delay_ms = 1000 // self.target_fps
        
        self._frame_processing = False 

        # --- IO 執行緒 (一般存檔) ---
        self.io_queue = queue.Queue(maxsize=5)
        self.io_thread = threading.Thread(target=self._io_worker_loop, daemon=True)
        self.io_thread.start()

        # --- 錄影執行緒 (專用 Queue) ---
        self.record_queue = queue.Queue(maxsize=30)
        self.record_thread = threading.Thread(target=self._record_worker_loop, daemon=True)
        self.record_thread.start()

        # --- 自動對焦變數 ---
        self.af_state = "IDLE" 
        self.motor_settle_timer = None
        self.af_algorithm = FocusAlgorithm(
            steps_config=config.FOCUS_STEPS_LIST,
            status_updater=lambda msg: self.status_updated.emit(msg)
        )
        self.af_image_count = 0
        self.af_folder = config.FOCUS_FOLDER

        # --- 採集協議變數 ---
        self.monitor_focus_enabled = False 
        self.reference_focus_score = 0.0   
        self.monitor_counter = 0           
        self.drift_threshold = 0.95        
        
        self.protocol_state = "IDLE"
        self.protocol_current_image = 0
        self.protocol_total_images = config.PROTOCOL_NUM_IMAGES
        self.protocol_started_by_user = False
        self.capture_filename_prefix = "" 
        
        self.protocol_n_side = 0          
        self.protocol_x_dir = 1           
        self.protocol_x_steps = 0         
        self.protocol_y_steps = 0         
        self.protocol_x_start_steps = 0   
        self.protocol_y_start_steps = 0   
        self.protocol_y_count = 0         
        self.center_offset_steps_x = 0
        self.center_offset_steps_y = 0

        # --- Z-Stack 掃描專用變數 ---
        self.z_stack_running = False
        self.z_stack_state = "IDLE"
        self.z_stack_folder = ""
        self.z_stack_log = []     
        self.z_stack_start_um = -2000.0
        self.z_stack_end_um = 2000.0
        self.z_stack_step_um = 20.0
        self.z_stack_current_dist = 0.0
        self.z_stack_counter = 0
        self.z_stack_total_steps = 0

    def _load_raw_unmix_tensor(self):
        try:
            if os.path.exists(config.UNMIX_MATRIX_PATH):
                data = np.load(config.UNMIX_MATRIX_PATH)
                if len(data.shape) == 4 and data.shape[2:] == (3, 3):
                    self.raw_unmix_tensor = data.astype(np.float32)
                    logging.info(f"✅ Unmix Tensor loaded. Shape: {data.shape}")
                else:
                    logging.warning(f"❌ Invalid Tensor Shape: {data.shape}.")
                    self.raw_unmix_tensor = None
            else:
                logging.warning(f"⚠️ Unmix Tensor not found at {config.UNMIX_MATRIX_PATH}")
                self.raw_unmix_tensor = None
        except Exception as e:
            logging.error(f"Failed to load Unmix Tensor: {e}")
            self.raw_unmix_tensor = None

    def _ensure_unmix_tensor_size(self, target_h, target_w):
        if self.raw_unmix_tensor is None:
            self.current_unmix_tensor = None
            return

        if (self.current_unmix_tensor is not None and 
            self.current_unmix_tensor.shape[0] == target_h and 
            self.current_unmix_tensor.shape[1] == target_w):
            return

        try:
            logging.info(f"Resizing Unmix Tensor to {target_w}x{target_h}...")
            raw_h, raw_w = self.raw_unmix_tensor.shape[:2]
            flat_tensor = self.raw_unmix_tensor.reshape(raw_h, raw_w, 9)
            resized_flat = cv2.resize(flat_tensor, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            self.current_unmix_tensor = resized_flat.reshape(target_h, target_w, 3, 3)
            logging.info("✅ Unmix Tensor resized successfully.")
        except Exception as e:
            logging.error(f"Error resizing unmix tensor: {e}")
            self.current_unmix_tensor = None

    def _io_worker_loop(self):
        logging.info("IO Worker thread started.")
        while self._is_running:
            try:
                task = self.io_queue.get(timeout=1.0)
                func, args = task
                try: func(*args)
                except Exception as e: logging.error(f"IO Worker Error: {e}")
                finally: self.io_queue.task_done()
            except queue.Empty: continue
            except Exception as e: logging.error(f"IO Loop Critical Error: {e}")

    def _record_worker_loop(self):
        logging.info("Record Worker thread started.")
        while self._is_running:
            try:
                frame_data = self.record_queue.get(timeout=1.0)
                if frame_data is None: continue 
                
                if self.video_writer is not None:
                    try:
                        self.video_writer.write(frame_data)
                    except Exception as e:
                        logging.error(f"Video Write Error: {e}")
                
                self.record_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Record Loop Critical Error: {e}")

    @Slot(str)
    def set_controls(self, controls_json_str):
        try:
            controls = json.loads(controls_json_str)
            with self._controls_lock:
                self._pending_controls = controls
        except Exception as e:
            logging.error(f"CameraWorker: Error in set_controls: {e}")

    @Slot(bool)
    def set_fluo_mode(self, enabled):
        self.is_fluo_mode = enabled
        mode_str = "Fluorescence (V6 + Unmix)" if enabled else "Bright Field (Global Variance)"
        logging.info(f"CameraWorker: Mode switched to {mode_str}")

    @Slot()
    def _capture_background_frame(self):
        try:
            logging.info("Capturing Background Frame (Raw)...")
            raw_buffer = self.picam2.capture_array("raw")
            raw_config = self.picam2.camera_configuration()["raw"]
            r_bg, g_bg, b_bg = get_raw_channels(raw_buffer, raw_config)
            self.bg_frame_raw = (r_bg, g_bg, b_bg)
            return True
        except Exception as e:
            logging.error(f"Failed to capture background frame: {e}")
            self.status_updated.emit("Error capturing background.")
            self.bg_frame_raw = None
            return False

    @Slot()
    def toggle_background_subtraction(self):
        if not self.bg_subtract_enabled:
            self.status_updated.emit("Capturing current view as BACKGROUND...")
            QMetaObject.invokeMethod(self, "_execute_bg_capture", Qt.QueuedConnection)
        else:
            self.bg_subtract_enabled = False
            self.bg_frame_raw = None
            self.status_updated.emit("Background Subtraction OFF.")

    @Slot()
    def _execute_bg_capture(self):
        if self._capture_background_frame():
            self.bg_subtract_enabled = True
            self.status_updated.emit("Background Captured. Subtraction ON.")
        else:
            self.bg_subtract_enabled = False
            self.status_updated.emit("Failed to capture background.")

    @Slot(bool)
    def set_focus_monitor(self, enabled):
        self.monitor_focus_enabled = enabled
        if enabled:
            self.monitor_counter = 0
            self.status_updated.emit("Focus Monitor: ON")
            if self.reference_focus_score == 0:
                 self.status_updated.emit("Focus Monitor: Please run AF once to set baseline.")
        else:
            self.status_updated.emit("Focus Monitor: OFF")

    @Slot()
    def _process_frame(self):
        if not self._is_running or self._frame_processing: return
        
        try:
            self._frame_processing = True 
            # 1. Controls
            pending = None
            with self._controls_lock:
                if self._pending_controls:
                    pending = self._pending_controls
                    self._pending_controls = None
            if pending:
                try: self.picam2.set_controls(pending)
                except Exception as e: logging.error(f"Failed to apply controls: {e}")

            # 2. Triggers
            if self._start_recording_triggered: self._start_recording_triggered = False; self._start_recording()
            if self._stop_recording_triggered: self._stop_recording_triggered = False; self._stop_recording()
            if self._awb_triggered: 
                self._awb_triggered = False; 
                self.status_updated.emit("Acquiring Flat Field...")
                self._run_awb()
            
            # 3. Capture
            try: raw_buffer = self.picam2.capture_array("raw")
            except Exception as e: self._frame_processing = False; return 
            if not self._is_running: self._frame_processing = False; return
            
            raw_config = self.picam2.camera_configuration()["raw"]
            r, g, b = get_raw_channels(raw_buffer, raw_config)
            
            self._ensure_unmix_tensor_size(r.shape[0], r.shape[1])
            
            # 4. Processing
            try: 
                if self.gain_maps is not None and self.is_fluo_mode:
                    tensor_to_use = self.current_unmix_tensor
                else:
                    tensor_to_use = None

                processed_rgb = apply_correction(
                    (r, g, b), 
                    self.gain_maps, 
                    self.bg_subtract_enabled,
                    self.bg_frame_raw, 
                    tensor_to_use 
                )
                
                if config.FLIP_VERTICAL: processed_rgb = cv2.flip(processed_rgb, 0)
                if config.FLIP_HORIZONTAL: processed_rgb = cv2.flip(processed_rgb, 1)
            except ValueError as e:
                logging.error(f"Processing error: {e}")
                processed_rgb = np.stack([r, g, b], axis=-1).astype(np.uint16)
            
            # 5. Output
            if self.video_writer is not None: 
                self._queue_video_frame(processed_rgb)

            q_img = convert_to_qimage(processed_rgb, self.ev_comp)
            if self._capture_triggered: 
                self._capture_triggered = False
                self._save_image_from_qimage(q_img)
            
            try:
                ptr = q_img.constBits()
                arr = np.array(ptr).reshape(q_img.height(), q_img.width(), 3)
                frame_to_encode_rgb = arr
                if self.stream_size and (self.stream_size[0] != arr.shape[1] or self.stream_size[1] != arr.shape[0]):
                    frame_to_encode_rgb = cv2.resize(arr, self.stream_size, interpolation=cv2.INTER_LINEAR)
                frame_8bit_bgr = cv2.cvtColor(frame_to_encode_rgb, cv2.COLOR_RGB2BGR)
                ret, jpeg_buffer = cv2.imencode(".jpg", frame_8bit_bgr)
                if ret: self.streaming_output.write(jpeg_buffer)
            except Exception as e: pass 
            
            # --- Z-Stack 收集邏輯 ---
            if self.z_stack_running:
                if self.z_stack_state == "WAIT_FOR_FRAME":
                    # 1. 儲存圖片 (轉為 8-bit 並套用 EV)
                    filename = f"img_{self.z_stack_counter:04d}.jpg"
                    full_path = os.path.join(self.z_stack_folder, filename)
                    try:
                        max_val = 65535.0
                        normalized = processed_rgb.astype(np.float32) / max_val
                        normalized *= self.ev_comp
                        np.clip(normalized, 0, 1, out=normalized) 
                        img_8bit = (normalized * 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(full_path, img_bgr)
                        
                        current_pos_um = self.z_stack_current_dist
                        self.z_stack_log.append((filename, current_pos_um))
                        self.status_updated.emit(f"Z-Stack [{self.z_stack_counter}/{self.z_stack_total_steps}]: {current_pos_um:.1f}um saved.")
                    except Exception as e:
                        logging.error(f"Save Error: {e}")
                    
                    if self.z_stack_counter >= self.z_stack_total_steps:
                        self._return_z_stack_center()
                    else:
                        self.z_stack_counter += 1
                        self.z_stack_current_dist += self.z_stack_step_um
                        steps = MotorConversion.um_to_microsteps_z(self.z_stack_step_um)
                        self.z_stack_state = "MOVING_NEXT"
                        self._issue_motor_command_for_z_stack(f"z{steps}")
                
                self._frame_processing = False
                return 

            # 6. AF Logic
            if self.af_state == "IDLE":
                if (self.monitor_focus_enabled and self.reference_focus_score > 100.0 and self.video_writer is None):
                    self.monitor_counter += 1
                    if self.monitor_counter > 30:
                        self.monitor_counter = 0
                        image_for_calc = processed_rgb[:, :, 1]
                        
                        if self.is_fluo_mode: 
                            curr_score = compute_score_fluo(image_for_calc)
                        else: 
                            # Monitor 模式預設使用 Tenengrad
                            curr_score = compute_score_tenengrad(image_for_calc)
                            
                        if curr_score < (self.reference_focus_score * self.drift_threshold):
                            self.status_updated.emit("Focus drift detected. Re-focusing...")
                            self.reference_focus_score = 0 
                            QMetaObject.invokeMethod(self, "start_drift_autofocus", Qt.QueuedConnection)
                self._frame_processing = False 
                return
                
            if self.af_state == "WAITING_FOR_MOTOR":
                self._frame_processing = False; return
                
            if self.af_state == "FOCUSING":
                image_for_calc = processed_rgb[:, :, 1]
                self.af_image_count += 1
                
                # --- 判斷粗/細調 ---
                current_step = self.af_algorithm.current_step_index
                total_steps = len(self.af_algorithm.steps_config)
                
                # 只有最後兩步 (N, N-1) 是細調，其餘為粗調
                is_coarse = (current_step <= total_steps - 2)
                
                metric = 0.0
                method_name = ""

                if self.is_fluo_mode:
                    if is_coarse:
                        # 螢光粗調使用 Variance
                        metric = compute_score_variance(image_for_calc) 
                        method_name = "Fluo-Coarse(Var)"
                    else:
                        # 螢光細調使用 Fluo (Morph)
                        metric = compute_score_fluo(image_for_calc) 
                        method_name = "Fluo-Fine(Morph)"
                else:
                    if is_coarse:
                        # 明視野粗調使用 Variance
                        metric = compute_score_variance(image_for_calc) 
                        method_name = "BF-Coarse(Var)"
                    else:
                        # 明視野細調使用 Tenengrad
                        metric = compute_score_tenengrad(image_for_calc) 
                        method_name = "BF-Fine(Tenengrad)"

                g_copy_u8 = cv2.normalize(image_for_calc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                self._queue_af_image_save(g_copy_u8, self.af_image_count)
                
                action = self.af_algorithm.step(metric)
                log_message = f"AF [{method_name} Step {self.af_image_count}]: {metric:.2e}"
                
                if action["type"] == "MOVE":
                    self.af_state = "WAITING_FOR_MOTOR"
                    self._issue_motor_command_for_af(action["command"])
                    log_message += f", Moving {action['command']}"
                elif action["type"] == "FINISHED":
                    if self.protocol_state == "PROTOCOL_AF_RUNNING":
                        self.af_state = "IDLE"; self.protocol_state = "PROTOCOL_CAPTURE"
                        self.status_updated.emit(log_message + ", Protocol AF Finished.")
                        self.reference_focus_score = metric 
                        QMetaObject.invokeMethod(self, "_run_protocol_step", Qt.QueuedConnection)
                    else:
                        self.af_state = "IDLE"
                        self.status_updated.emit(log_message + ", Finished.")
                        self.af_status_finished.emit() 
                        self.reference_focus_score = metric
                    self._frame_processing = False; return 
                elif action["type"] == "WAIT":
                    log_message += ", Waiting..." 
                self.status_updated.emit(log_message) 
        finally:
            self._frame_processing = False 

    # --- Motor & Protocol Wrappers ---
    def _issue_motor_command_for_af(self, command):
        threading.Thread(target=self.motor_control.send_command, args=(command,), kwargs={"silent": True}).start()
        if self.motor_settle_timer is None:
            self.motor_settle_timer = QTimer(self); self.motor_settle_timer.setSingleShot(True)
            self.motor_settle_timer.timeout.connect(self._on_af_motor_settled)
        self.motor_settle_timer.start(150) 
    
    @Slot()
    def _on_af_motor_settled(self):
        if self.af_state == "WAITING_FOR_MOTOR": self.af_state = "FOCUSING"

    def _issue_protocol_motor_command(self, command):
        self.af_state = "WAITING_FOR_MOTOR"
        threading.Thread(target=self._protocol_motor_thread_target, args=(command,)).start()

    def _protocol_motor_thread_target(self, command):
        self.motor_control.send_command(command, silent=True)
        QMetaObject.invokeMethod(self, "_on_protocol_motor_done", Qt.QueuedConnection)

    @Slot()
    def _on_protocol_motor_done(self):
        if self.af_state == "WAITING_FOR_MOTOR":
            self.af_state = "IDLE"; self._run_protocol_step()

    @Slot(str)
    def _queue_capture(self, filename_prefix):
        self._capture_triggered = True; self.capture_filename_prefix = filename_prefix

    @Slot()
    def start_autofocus(self):
        if self.af_state == "IDLE":
            self.af_algorithm.steps_config = config.FOCUS_STEPS_LIST; self._start_af_common()
    @Slot()
    def start_drift_autofocus(self):
        if self.af_state == "IDLE":
            self.af_algorithm.steps_config = config.FOCUS_STEPS_DRIFT; self._start_af_common()
    
    def _start_af_common(self):
        try:
            if os.path.exists(self.af_folder):
                files = glob.glob(os.path.join(self.af_folder, "*"))
                for f in files:
                    try: os.remove(f)
                    except Exception: pass
            else:
                os.makedirs(self.af_folder, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to clear AF folder: {e}")

        self.af_image_count = 0
        self.af_algorithm.start()
        self.af_state = "FOCUSING"
        self._process_frame() 

    @Slot()
    def cancel_autofocus(self):
        if self.af_state != "IDLE":
            if self.motor_settle_timer and self.motor_settle_timer.isActive(): self.motor_settle_timer.stop()
            self.af_algorithm.cancel(); self.af_state = "IDLE"; self.af_status_finished.emit()

    # --- Z-Stack 相關功能 ---
    @Slot()
    def start_z_stack_collection(self):
        if self.z_stack_running or self.af_state != "IDLE":
            logging.warning("Cannot start Z-Stack: System busy.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.z_stack_folder = os.path.join(config.CAPTURE_FOLDER, f"Z_Stack_{timestamp}")
        os.makedirs(self.z_stack_folder, exist_ok=True)
        logging.info(f"Starting Z-Stack Collection. Saving to: {self.z_stack_folder}")
        
        self.z_stack_running = True
        self.z_stack_log = []
        self.z_stack_counter = 0
        
        start_offset = -2000
        total_range = 4000
        step_size = 20
        
        self.z_stack_step_um = step_size
        self.z_stack_total_steps = int(total_range / step_size)
        self.z_stack_current_dist = start_offset 

        self.z_stack_state = "MOVING_TO_START"
        steps = MotorConversion.um_to_microsteps_z(start_offset)
        
        self.status_updated.emit(f"Z-Stack: Moving to start ({start_offset}um)...")
        self._issue_motor_command_for_z_stack(f"z{steps}")

    def _issue_motor_command_for_z_stack(self, command):
        threading.Thread(target=self._z_stack_motor_thread, args=(command,)).start()

    def _z_stack_motor_thread(self, command):
        self.motor_control.send_command(command, silent=True)
        time.sleep(0.4) 
        QMetaObject.invokeMethod(self, "_on_z_stack_motor_settled", Qt.QueuedConnection)

    @Slot()
    def _on_z_stack_motor_settled(self):
        if not self.z_stack_running: return

        if self.z_stack_state == "MOVING_TO_START":
            self.z_stack_state = "WAIT_FOR_FRAME" 
            
        elif self.z_stack_state == "MOVING_NEXT":
            self.z_stack_state = "WAIT_FOR_FRAME" 
            
        elif self.z_stack_state == "RETURNING":
            self.status_updated.emit(f"Z-Stack Complete. Saved {len(self.z_stack_log)} images.")
            self._save_z_stack_csv() 
            self.z_stack_running = False
            self.z_stack_state = "IDLE"

    def _save_z_stack_csv(self):
        csv_path = os.path.join(self.z_stack_folder, "data_log.csv")
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Position_um'])
                writer.writerows(self.z_stack_log)
            logging.info(f"Z-Stack log saved: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to save CSV: {e}")

    def _return_z_stack_center(self):
        return_dist = -self.z_stack_current_dist
        steps = MotorConversion.um_to_microsteps_z(return_dist)
        self.status_updated.emit("Z-Stack: Returning to center...")
        self.z_stack_state = "RETURNING"
        self._issue_motor_command_for_z_stack(f"z{steps}")
            
    # --- Protocol Logic ---
    @Slot(str, str, str) 
    def start_acquisition_protocol(self, grid_n_str, range_um_str, mode_str):
        if self.af_state == "IDLE" and self.protocol_state == "IDLE":
            try:
                n_side = int(grid_n_str)
                range_um = float(range_um_str)
                num_images = n_side * n_side
                if n_side < 2: raise ValueError("Grid size must be >= 2")
                
                self.protocol_total_images = num_images
                self.protocol_n_side = n_side
                self.protocol_mode = mode_str 
            except ValueError as e: 
                self.status_updated.emit(f"Error: {e}")
                return

            step_um = range_um / (n_side - 1)
            logging.info(f"Protocol Start: Mode={mode_str}, Grid={n_side}x{n_side}, Range={range_um}um, Step={step_um:.2f}um")

            self.protocol_x_steps = MotorConversion.um_to_microsteps_xy(step_um) 
            self.protocol_y_steps = -MotorConversion.um_to_microsteps_xy(step_um) 
            start_move_um = range_um / 2.0
            self.protocol_x_start_steps = -MotorConversion.um_to_microsteps_xy(start_move_um) 
            self.protocol_y_start_steps = MotorConversion.um_to_microsteps_xy(start_move_um) 
            
            self.protocol_current_image = 0
            self.protocol_started_by_user = True
            self.protocol_y_count = 0
            self.protocol_x_dir = 1 
            self.center_offset_steps_x = 0
            self.center_offset_steps_y = 0

            self.protocol_state = "PROTOCOL_MOVE_TO_CENTER" 
            self.status_updated.emit(f"Protocol: Moving to Center (Range: {range_um}um)...")
            self._issue_protocol_motor_command(f"sx{self.center_offset_steps_x}")

    @Slot()
    def cancel_acquisition_protocol(self):
        if self.protocol_state != "IDLE":
            return_x_steps = -self.center_offset_steps_x; return_y_steps = -self.center_offset_steps_y
            threading.Thread(target=self.motor_control.send_command, args=(f"sx{return_x_steps}",), kwargs={"silent": True}).start()
            threading.Thread(target=self.motor_control.send_command, args=(f"sy{return_y_steps}",), kwargs={"silent": True}).start()
            self.center_offset_steps_x = 0; self.center_offset_steps_y = 0
            self.protocol_state = "IDLE"; self.af_algorithm.cancel(); self.af_state = "IDLE"
            self.protocol_started_by_user = False; self.status_updated.emit("Protocol: Cancelled.")
            if self.motor_settle_timer: self.motor_settle_timer.stop()
                
    @Slot()
    def _run_protocol_step(self):
        if self.protocol_state == "PROTOCOL_MOVE_TO_CENTER_END":
            self.protocol_state = "PROTOCOL_MOVE_TO_CENTER_END_Y"
            return_y_steps = -self.center_offset_steps_y; self._issue_protocol_motor_command(f"sy{return_y_steps}"); return
        elif self.protocol_state == "PROTOCOL_MOVE_TO_CENTER_END_Y":
            self.protocol_state = "IDLE"; self.protocol_started_by_user = False
            self.center_offset_steps_x = 0; self.center_offset_steps_y = 0
            self.status_updated.emit("Protocol: Complete."); return
        
        if self.protocol_current_image >= self.protocol_total_images and self.protocol_state == "PROTOCOL_CAPTURE":
            self.protocol_state = "PROTOCOL_MOVE_TO_CENTER_END"
            self.status_updated.emit("Protocol: Finished. Returning..."); return_x_steps = -self.center_offset_steps_x
            self._issue_protocol_motor_command(f"sx{return_x_steps}"); return

        elif self.protocol_state == "PROTOCOL_MOVE_TO_CENTER":
            self.protocol_state = "PROTOCOL_MOVE_TO_CENTER_Y"; self._issue_protocol_motor_command(f"sy{self.center_offset_steps_y}")
        elif self.protocol_state == "PROTOCOL_MOVE_TO_CENTER_Y":
            self.protocol_state = "PROTOCOL_MOVE_TO_START_X"; self.status_updated.emit("Protocol: Moving to Start...")
            self._issue_protocol_motor_command(f"sx{self.protocol_x_start_steps}")
        elif self.protocol_state == "PROTOCOL_MOVE_TO_START_X":
             self.protocol_state = "PROTOCOL_MOVE_TO_START_Y"; self._issue_protocol_motor_command(f"sy{self.protocol_y_start_steps}")
        elif self.protocol_state == "PROTOCOL_MOVE_TO_START_Y":
            self.center_offset_steps_x = self.protocol_x_start_steps; self.center_offset_steps_y = self.protocol_y_start_steps
            self.protocol_state = "PROTOCOL_WAIT_AF"; QMetaObject.invokeMethod(self, "_run_protocol_step", Qt.QueuedConnection); return
        
        elif self.protocol_state == "PROTOCOL_WAIT_AF":
            self.status_updated.emit(f"Protocol {self.protocol_current_image+1}/{self.protocol_total_images}: AF...")
            
            if hasattr(self, 'protocol_mode') and self.protocol_mode == 'stitching':
                QMetaObject.invokeMethod(self, "start_autofocus", Qt.QueuedConnection)
            else:
                QMetaObject.invokeMethod(self, "start_drift_autofocus", Qt.QueuedConnection)
            
            self.protocol_state = "PROTOCOL_AF_RUNNING"
        
        elif self.protocol_state == "PROTOCOL_CAPTURE":
            self.protocol_current_image += 1
            n = self.protocol_n_side
            row_index = self.protocol_y_count
            row_num = row_index + 1
            col_index_linear = (self.protocol_current_image - 1) % n 
            col_num = col_index_linear + 1
            
            filename = f"xy_R{row_num:02d}_C{col_num:02d}"
            self._queue_capture(filename); self.status_updated.emit(f"Protocol: Capturing {filename}...")
            
            if self.protocol_current_image >= self.protocol_total_images: 
                QMetaObject.invokeMethod(self, "_run_protocol_step", Qt.QueuedConnection)
                return 
            
            if self.protocol_current_image % n == 0: 
                self.protocol_state = "PROTOCOL_MOVE_Y" 
            else: 
                self.protocol_state = "PROTOCOL_MOVE_X" 
            
            QMetaObject.invokeMethod(self, "_run_protocol_step", Qt.QueuedConnection)
        
        elif self.protocol_state == "PROTOCOL_MOVE_X":
            self.protocol_state = "PROTOCOL_WAIT_AF"
            steps = self.protocol_x_steps 
            self.center_offset_steps_x += steps
            self._issue_protocol_motor_command(f"sx{steps}")

        elif self.protocol_state == "PROTOCOL_MOVE_Y":
            self.protocol_state = "PROTOCOL_REWIND_X" 
            self.protocol_y_count += 1
            steps_y = self.protocol_y_steps
            self.center_offset_steps_y += steps_y
            self._issue_protocol_motor_command(f"sy{steps_y}")

        elif self.protocol_state == "PROTOCOL_REWIND_X": 
            self.protocol_state = "PROTOCOL_WAIT_AF"
            n = self.protocol_n_side
            steps_rewind = -1 * self.protocol_x_steps * (n - 1)
            self.center_offset_steps_x += steps_rewind
            self._issue_protocol_motor_command(f"sx{steps_rewind}")
            
    def _queue_af_image_save(self, image_u8, count):
        try: self.io_queue.put_nowait((self._perform_disk_write, (image_u8, count)))
        except queue.Full: pass
    def _perform_disk_write(self, image_u8, count):
        try: cv2.imwrite(os.path.join(self.af_folder, f"image_{count:03d}.jpg"), image_u8)
        except Exception: pass
    
    @Slot()
    def run(self):
        self.frame_timer = QTimer(self); self.frame_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_timer.setInterval(self.target_delay_ms); self.frame_timer.timeout.connect(self._process_frame)
        self.frame_timer.start()
    @Slot()
    def stop(self):
        self._is_running = False
        if self.frame_timer: self.frame_timer.stop()
        if self.video_writer: self._stop_recording()
        if self.thread(): self.thread().quit() 
    
    def _run_awb(self):
        self.status_updated.emit("Calculating Flat Field..."); 
        try:
            raw_config = self.picam2.camera_configuration()["raw"]
            self.flat_field_channels = get_raw_channels(self.picam2.capture_array("raw"), raw_config)
            self.gain_maps = calculate_gain_maps(self.flat_field_channels)
            if self.gain_maps is None: self.status_updated.emit("Error: Too dark!"); self.mode = "live_auto"
            else:
                self.mode = "live_corrected"
                msg = "AWB Applied."
                if self.current_unmix_tensor is not None and self.is_fluo_mode: msg += " (Unmix ON)"
                self.status_updated.emit(msg)
        except Exception as e:
            logging.error(f"AWB failed: {e}"); self.status_updated.emit(f"AWB Error: {e}"); self.gain_maps = None
        
    def _save_image_from_qimage(self, q_img):
        if self.protocol_started_by_user and hasattr(self, 'capture_filename_prefix'): target_folder = config.CAPTURE_FOLDER
        else: target_folder = "." 
        os.makedirs(target_folder, exist_ok=True)
        if self.protocol_started_by_user and hasattr(self, 'capture_filename_prefix'):
            filename = os.path.join(target_folder, f"{self.capture_filename_prefix}_{datetime.now().strftime('%H%M%S')}.png")
            self.capture_filename_prefix = "" 
        else: filename = os.path.join(target_folder, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        try:
            ptr = q_img.constBits(); h, w = q_img.height(), q_img.width()
            arr = np.array(ptr).reshape(h, w, 3); bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr); self.status_updated.emit(f"Saved: {filename}")
        except Exception as e: self.status_updated.emit(f"Error saving image: {e}")
            
    def _start_recording(self):
        if self.video_writer is not None: return
        filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.video_fps, self.video_size)
        self.status_updated.emit(f"🔴 Recording: {filename}")
    
    def _stop_recording(self):
        if self.video_writer is None: return
        self.video_writer.release(); self.video_writer = None; self.status_updated.emit("Recording Stopped.")

    def _queue_video_frame(self, rgb_16bit_array):
        try:
            max_val = 65535.0
            normalized = rgb_16bit_array.astype(np.float32) / max_val
            normalized *= self.ev_comp; np.clip(normalized, 0, 1, out=normalized)
            frame_8bit_rgb = (normalized * 255).astype(np.uint8)
            frame_resized_rgb = cv2.resize(frame_8bit_rgb, self.video_size, interpolation=cv2.INTER_LINEAR)
            frame_8bit_bgr = cv2.cvtColor(frame_resized_rgb, cv2.COLOR_RGB2BGR)
            self.record_queue.put_nowait(frame_8bit_bgr)
        except queue.Full:
            logging.warning("Video Queue Full! Dropping frame.")
        except Exception as e:
            logging.error(f"Error queueing video frame: {e}")