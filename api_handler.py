# api_handler.py
import logging
import json
import threading
import time
import os       
import shutil   
from PySide6.QtCore import QMetaObject, Qt, Q_ARG

import config 

class ApiHandler:
    def __init__(self, camera_manager, motor_control, light_control, server_instance):
        self.cam_manager = camera_manager
        self.motor = motor_control
        self.light = light_control
        self.server = server_instance 

    def _force_clear_folder(self, folder_path):
        try:
            if os.path.exists(folder_path): shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True); return True
        except Exception as e: logging.error(f"Failed to clear folder: {e}"); return False

    def handle_request(self, cmd, params):
        cam_worker = self.cam_manager.camera_worker
        
        # (Status)
        current_gain = 1.0; current_exp_us = 10000; current_ae_enabled = True 
        try:
            if self.cam_manager.picam2.started:
                current_metadata = self.cam_manager.picam2.capture_metadata()
                current_gain = current_metadata.get("AnalogueGain", 1.0)
                current_exp_us = current_metadata.get("ExposureTime", 10000)
            if self.cam_manager.persistent_locked_controls:
                current_ae_enabled = False
                current_gain = self.cam_manager.persistent_locked_controls.get("AnalogueGain", current_gain)
                current_exp_us = self.cam_manager.persistent_locked_controls.get("ExposureTime", current_exp_us)
        except Exception: pass
        
        status_msg = "OK"
        response_data = {
            "message": status_msg, "ev": None, "af_finished": False, 
            "current_res_key": self.cam_manager.current_res_key, 
            "current_fps_key": self.cam_manager.current_fps_key,
            "current_gain": current_gain, "current_exp_us": current_exp_us, 
            "ae_enabled": current_ae_enabled, "bg_subtract_enabled": False, "monitor_enabled": False 
        }
        if cam_worker:
             response_data["bg_subtract_enabled"] = cam_worker.bg_subtract_enabled
             response_data["monitor_enabled"] = cam_worker.monitor_focus_enabled 
        
        is_af_running = self.cam_manager.af_state_running
        
        if cam_worker is None and cmd not in ['apply_settings', 'get_status', 'exit_app']: 
            response_data["message"] = "Error: Camera busy."
        else:
            try:
                if cam_worker is not None: 
                    if cmd == 'awb': cam_worker._awb_triggered = True; status_msg = "AWB Triggered"
                    elif cmd == 'ev_up': cam_worker.ev_comp *= 1.2; status_msg = f"EV: {cam_worker.ev_comp:.2f}x"
                    elif cmd == 'ev_down': cam_worker.ev_comp /= 1.2; status_msg = f"EV: {cam_worker.ev_comp:.2f}x"
                    elif cmd == 'toggle_bg_subtract': QMetaObject.invokeMethod(cam_worker, "toggle_background_subtraction", Qt.QueuedConnection); status_msg = "Toggle BG subtract."
                    elif cmd == 'toggle_monitor':
                        enabled = params.get('enabled', ['false'])[0].lower() == 'true'
                        QMetaObject.invokeMethod(cam_worker, "set_focus_monitor", Qt.QueuedConnection, Q_ARG(bool, enabled))
                        status_msg = f"Monitor: {'ON' if enabled else 'OFF'}"
                    elif cmd == 'capture': cam_worker._capture_triggered = True; status_msg = "Capture Triggered"
                    elif cmd == 'record':
                        if cam_worker.video_writer is None: cam_worker._start_recording_triggered = True; status_msg = "Rec Started"
                        else: cam_worker._stop_recording_triggered = True; status_msg = "Rec Stopped"
                
                if cmd == 'light_bf':
                    self.light.set_light(self.light.bf_pin, not self.server.bf_light_state)
                    self.server.bf_light_state = not self.server.bf_light_state
                    status_msg = f"BF {'ON' if self.server.bf_light_state else 'OFF'}"
                elif cmd == 'light_fluo':
                    new_state = not self.server.fluo_light_state
                    self.light.set_light(self.light.fluo_pin, new_state)
                    self.server.fluo_light_state = new_state
                    # 【新增】通知 Worker 切換 AF 模式
                    if cam_worker is not None:
                        QMetaObject.invokeMethod(cam_worker, "set_fluo_mode", Qt.QueuedConnection, Q_ARG(bool, new_state))
                    status_msg = f"Fluo {'ON' if new_state else 'OFF'}"

                elif cmd.startswith('move_'):
                    if is_af_running and cmd.startswith('move_z_'): status_msg = "Error: AF running."
                    else:
                        value = params.get('value', [0])[0]; parts = cmd.split('_'); axis = parts[1]; direction = int(parts[2]); steps = int(value)
                        command = f"s{axis}{steps * direction}"
                        threading.Thread(target=self.motor.send_command, args=(command,)).start()
                        status_msg = f"Move: {command}"
                
                can_set_controls = (cam_worker is not None)
                if cmd == 'aec_lock':
                    if can_set_controls:
                        metadata = self.cam_manager.picam2.capture_metadata()
                        exposure = metadata.get("ExposureTime", 10000); gain = metadata.get("AnalogueGain", 1.0)
                        duration = max(exposure, metadata.get("FrameDuration", exposure))
                        controls = {"AeEnable": False, "ExposureTime": exposure, "AnalogueGain": gain, "FrameDurationLimits": (int(duration), int(duration))}
                        QMetaObject.invokeMethod(cam_worker, "set_controls", Qt.QueuedConnection, Q_ARG(str, json.dumps(controls)))
                        QMetaObject.invokeMethod(self.cam_manager, "on_controls_locked", Qt.QueuedConnection, Q_ARG(str, json.dumps(controls)))
                        response_data["ae_enabled"] = False; response_data["current_gain"] = gain; response_data["current_exp_us"] = exposure
                        status_msg = "Exposure Locked."

                elif cmd == 'set_manual_exposure':
                    if can_set_controls:
                        gain = params.get('gain', [1.0])[0]; exp_ms = params.get('exp', [100])[0]; exp_us = int(float(exp_ms)) * 1000
                        controls = {"AeEnable": False, "AnalogueGain": float(gain), "ExposureTime": exp_us, "FrameDurationLimits": (exp_us, exp_us)}
                        QMetaObject.invokeMethod(cam_worker, "set_controls", Qt.QueuedConnection, Q_ARG(str, json.dumps(controls)))
                        QMetaObject.invokeMethod(self.cam_manager, "on_controls_locked", Qt.QueuedConnection, Q_ARG(str, json.dumps(controls)))
                        response_data["ae_enabled"] = False; response_data["current_gain"] = float(gain); response_data["current_exp_us"] = exp_us
                        status_msg = "Manual Exp Set."

                elif cmd == 'reset_auto':
                    if can_set_controls:
                        default_fps = config.FPS_OPTIONS.get(self.cam_manager.current_fps_key, 30)
                        min_dur = int(1_000_000 / default_fps)
                        controls = {"AeEnable": True, "AwbEnable": True, "FrameDurationLimits": (min_dur, 1000000)}
                        QMetaObject.invokeMethod(cam_worker, "set_controls", Qt.QueuedConnection, Q_ARG(str, json.dumps(controls)))
                        QMetaObject.invokeMethod(self.cam_manager, "clear_locked_controls", Qt.QueuedConnection)
                        response_data["ae_enabled"] = True; status_msg = "Auto Exp Set."
                
                elif cmd == 'autofocus':
                    if is_af_running: QMetaObject.invokeMethod(self.cam_manager, "cancel_autofocus", Qt.QueuedConnection); status_msg = "AF Cancelled."
                    else:
                        if self.cam_manager.af_status["finished"]: self.cam_manager.af_status["finished"] = False 
                        QMetaObject.invokeMethod(self.cam_manager, "start_autofocus", Qt.QueuedConnection); status_msg = "AF Started..."
                
                elif cmd == 'start_protocol':
                    self._force_clear_folder(config.CAPTURE_FOLDER)
                    num_images = params.get('num_images', ['25'])[0]
                    QMetaObject.invokeMethod(self.cam_manager.camera_worker, "start_acquisition_protocol", Qt.QueuedConnection, Q_ARG(str, num_images))
                    status_msg = f"Protocol ({num_images}) Started."

                elif cmd == 'cancel_protocol':
                    QMetaObject.invokeMethod(self.cam_manager.camera_worker, "cancel_acquisition_protocol", Qt.QueuedConnection); status_msg = "Protocol Cancelled."
                elif cmd == 'exit_app':
                    status_msg = "Shutting down..."; threading.Thread(target=lambda: (time.sleep(1), self.server.app.quit())).start()
                elif cmd == 'get_status': status_msg = "Status refreshed."
                elif cmd == 'apply_settings':
                    res = params.get('res', [None])[0]; fps = params.get('fps', [None])[0]
                    if res and fps: QMetaObject.invokeMethod(self.cam_manager, "restart_camera_system", Qt.QueuedConnection, Q_ARG(str, res), Q_ARG(str, fps)); status_msg = "Restarting..."
                    else: status_msg = "Settings error."

            except Exception as e: logging.error(f"Cmd Error '{cmd}': {e}"); status_msg = f"Error: {e}"
        
        if cam_worker: response_data["ev"] = f"{cam_worker.ev_comp:.2f}x"
        else: response_data["ev"] = f"{self.cam_manager.persistent_ev_comp:.2f}x"
        response_data["message"] = status_msg
        
        if self.cam_manager.af_status["finished"]:
            response_data["af_finished"] = True; self.cam_manager.af_status["finished"] = False 
            
        return response_data