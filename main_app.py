# main_app.py
import sys, logging, threading, io, socketserver, time, json 
from http import server
from threading import Condition
from urllib.parse import parse_qs, urlparse
from picamera2 import Picamera2
from PySide6.QtCore import (QCoreApplication, QObject, QThread, Signal, Slot,
                              QMetaObject, Qt, Q_ARG)
from image_processing import calculate_gain_maps
import os # 【已修正】導入 os 模組
try:
    from skimage.transform import resize
except ImportError:
    logging.error("ImportError: scikit-image not found...")
    sys.exit(1)
import config
from hardware_control import LightControl, MotorControl
from workers import CameraWorker

from api_handler import ApiHandler

class StreamingOutput(io.BufferedIOBase):
    def __init__(self): self.frame = None; self.condition = Condition()
    def write(self, buf):
        with self.condition: self.frame = buf; self.condition.notify_all()

# --- (Python 後端 - CameraManager 類別不變) ---
class CameraManager(QObject):
    status_updated = Signal(str)
    
    def __init__(self, picam2, output, motor_control, app):
        super().__init__()
        self.picam2 = picam2
        self.output = output
        self.motor_control = motor_control
        self.app = app
        
        self.camera_thread = None; self.camera_worker = None
        self.current_res_key = config.DEFAULT_RESOLUTION_KEY; self.current_fps_key = config.DEFAULT_FPS_KEY
        self.persistent_flat_field_channels = None; self.persistent_mode = "live_auto"
        self.persistent_ev_comp = 1.0; 
        self.persistent_locked_controls = {} 
        
        self.af_state_running = False
        self.af_status = {"finished": False}
    
    @Slot(str)
    def on_controls_locked(self, controls_json):
        try:
            controls = json.loads(controls_json) 
            if "AeEnable" in controls and controls["AeEnable"] == False:
                logging.info(f"CameraManager: Persisting locked controls: {controls}")
                self.persistent_locked_controls.update(controls)
            else:
                logging.warning(f"CameraManager: on_controls_locked called with AeEnable=True. Ignoring persistence.")
        except json.JSONDecodeError as e:
            logging.error(f"CameraManager: Failed to parse controls_json: {e}")

    @Slot()
    def clear_locked_controls(self):
        logging.info("CameraManager: Clearing persistent locked controls.")
        self.persistent_locked_controls = {}

    def _persist_current_state(self):
        if not self.picam2.started:
            logging.info("Camera not started. Skipping state persistence.")
            self.persistent_locked_controls = {} 
            return
            
        if self.camera_worker:
            self.persistent_flat_field_channels = self.camera_worker.flat_field_channels
            self.persistent_mode = self.camera_worker.mode
            self.persistent_ev_comp = self.camera_worker.ev_comp
        
        try:
            meta = self.picam2.capture_metadata()
            ae_enabled = meta.get("AeEnable", True)
            
            if ae_enabled:
                logging.info("AEC is enabled. Clearing persistent controls.")
                self.persistent_locked_controls = {} 
            else:
                self.persistent_locked_controls = {
                    "AeEnable": False,
                    "AnalogueGain": meta.get("AnalogueGain", 1.0),
                    "ExposureTime": meta.get("ExposureTime", 10000),
                    "AwbEnable": meta.get("AwbEnable", True),
                    "ColourGains": meta.get("ColourGains", (1.0, 1.0)),
                    "FrameDurationLimits": (meta.get("FrameDuration", 10000), meta.get("FrameDuration", 10000))
                }
                logging.info(f"AEC is off. Persisted controls: {self.persistent_locked_controls}")

        except Exception as e:
            logging.warning(f"Could not read metadata to persist controls: {e}")
            self.persistent_locked_controls = {}

    @Slot(str, str)
    def restart_camera_system(self, res_key, fps_key):
        self.status_updated.emit(f"Stopping camera...")
        
        self._persist_current_state()

        if self.camera_worker and self.camera_thread:
            QMetaObject.invokeMethod(self.camera_worker, "stop", Qt.QueuedConnection)
            if not self.camera_thread.wait(3000): 
                 logging.warning("CameraWorker thread did not quit in time.")
            logging.info("CameraWorker thread successfully stopped (for restart).")

        if self.picam2.started:
            try:
                self.picam2.stop()
                logging.info("picam2 stopped.")
            except Exception as e:
                logging.warning(f"Error stopping picam2 (continuing...): {e}")

        self.camera_worker = None; self.camera_thread = None
        self.status_updated.emit(f"Reconfiguring camera to {res_key} @ {fps_key} FPS...")
        try:
            res_tuple = config.RESOLUTION_OPTIONS.get(res_key, config.RESOLUTION_OPTIONS[config.DEFAULT_RESOLUTION_KEY])
            fps_val = config.FPS_OPTIONS.get(fps_key, config.FPS_OPTIONS[config.DEFAULT_FPS_KEY])
            req_raw_w, req_raw_h = res_tuple
            req_debayered_w = req_raw_w // 2; req_debayered_h = req_raw_h // 2
            
            cam_config = self.picam2.create_preview_configuration(raw={"size": (req_raw_w, req_raw_h), "format": "SBGGR16"})
            self.picam2.configure(cam_config)
            
            actual_config = self.picam2.camera_configuration()
            actual_raw_size = actual_config["raw"]["size"] 
            actual_raw_w, actual_raw_h = actual_raw_size
            actual_debayered_w = actual_raw_w // 2; actual_debayered_h = actual_raw_h // 2
            if (actual_raw_w, actual_raw_h) != res_tuple:
                logging.warning(f"Camera resolution adjusted: Requested {res_tuple}, Got {(actual_raw_w, actual_raw_h)}")
            
            if self.persistent_locked_controls:
                logging.info(f"Re-applying persistent controls: {self.persistent_locked_controls}")
                self.picam2.set_controls(self.persistent_locked_controls)
            else:
                logging.info("No persistent controls, ensuring AEC/AWB are enabled.")
                req_fps = config.FPS_OPTIONS.get(fps_key, 30)
                default_min_duration = int(1_000_000 / req_fps)
                default_max_duration = 1000000 
                self.picam2.set_controls({
                    "AeEnable": True, 
                    "AwbEnable": True,
                    "FrameDurationLimits": (default_min_duration, default_max_duration)
                })
                
            self.current_res_key = res_key; self.current_fps_key = fps_key
            
            self.camera_thread = QThread()
            self.camera_worker = CameraWorker(self.picam2, self.output, self.motor_control)
            
            self.camera_worker.ev_comp = self.persistent_ev_comp; self.camera_worker.mode = self.persistent_mode
            
            if self.persistent_flat_field_channels is not None:
                self.status_updated.emit("Recalculating gain maps for new resolution...")
                try:
                    new_shape = (actual_debayered_h, actual_debayered_w) 
                    r_flat_old, g_flat_old, b_flat_old = self.persistent_flat_field_channels
                    logging.info(f"Resizing flat-field from {r_flat_old.shape} to {new_shape}")
                    r_flat_new = resize(r_flat_old, new_shape, anti_aliasing=True, preserve_range=True)
                    g_flat_new = resize(g_flat_old, new_shape, anti_aliasing=True, preserve_range=True)
                    b_flat_new = resize(b_flat_old, new_shape, anti_aliasing=True, preserve_range=True)
                    resized_flat_field_channels = (r_flat_new, g_flat_new, b_flat_new)
                    self.camera_worker.gain_maps = calculate_gain_maps(resized_flat_field_channels)
                    self.camera_worker.flat_field_channels = resized_flat_field_channels
                    self.camera_worker.mode = "live_corrected"
                    self.status_updated.emit("Gain maps recalculated successfully.")
                except Exception as e:
                    self.status_updated.emit(f"Error recalculating gain maps: {e}"); logging.error(f"Failed to recalculate gain maps: {e}")
                    self.camera_worker.mode = "live_auto"
            
            self.camera_worker.target_fps = fps_val
            self.camera_worker.stream_size = (req_debayered_w, req_debayered_h) 
            self.camera_worker.video_size = (req_debayered_w, req_debayered_h)

            self.camera_worker.moveToThread(self.camera_thread)
            self.camera_worker.status_updated.connect(self.status_updated)
            self.camera_worker.controls_locked.connect(self.on_controls_locked)
            self.camera_worker.af_status_finished.connect(self._on_af_finished)
            
            self.camera_thread.started.connect(self.camera_worker.run)
            
            self.picam2.start(); self.camera_thread.start()
            self.status_updated.emit(f"Camera restarted (Sensor: {actual_debayered_w}x{actual_debayered_h}, Stream: {req_debayered_w}x{req_debayered_h} @ {fps_val} FPS)")
            
        except Exception as e:
            self.status_updated.emit(f"Error restarting camera: {e}"); logging.error(f"Failed to restart camera: {e}")
    
    @Slot()
    def _on_af_finished(self):
        logging.info("CameraManager: Received AF finished signal from worker.")
        self.af_state_running = False
        self.af_status["finished"] = True

    @Slot()
    def start_autofocus(self):
        if self.af_state_running:
            logging.warning("CameraManager: Start AF called, but AF is already running.")
            return
            
        if not self.camera_worker:
            logging.error("CameraManager: Cannot start AF, CameraWorker is not running.")
            return

        logging.info("CameraManager: Requesting worker to start AF state machine.")
        self.af_state_running = True
        self.af_status["finished"] = False
        QMetaObject.invokeMethod(self.camera_worker, "start_autofocus", Qt.QueuedConnection)

    @Slot()
    def cancel_autofocus(self):
        if not self.af_state_running:
            logging.warning("CameraManager: Cancel AF called, but AF is not running.")
            return

        if not self.camera_worker:
            logging.error("CameraManager: Cannot cancel AF, CameraWorker is not running.")
            return

        logging.info("CameraManager: Requesting worker to cancel AF state machine.")
        QMetaObject.invokeMethod(self.camera_worker, "cancel_autofocus", Qt.QueuedConnection)


# --- (StreamingHandler 類別不變) ---
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/': 
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
            
        elif url.path == '/index.html':
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read().encode('utf-8')
                
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', len(content))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                self.wfile.write(content)
            
            except FileNotFoundError:
                logging.error("index.html not found.")
                self.send_error(404, "File Not Found: index.html")
            except Exception as e:
                logging.error(f"Error reading index.html: {e}")
                self.send_error(500, f"Server Error: {e}")
            
        elif url.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with self.server.output.condition:
                        self.server.output.condition.wait()
                        frame = self.server.output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                if "Broken pipe" not in str(e):
                    logging.warning('Streaming client removed: %s', str(e))
        elif url.path == '/control':
            self.handle_control_request(url)
        else:
            self.send_error(404)
            self.end_headers()
        
    def handle_control_request(self, url):
        try:
            params = parse_qs(url.query)
            cmd = params.get('cmd', [None])[0]
            
            response_data = self.server.api_handler.handle_request(cmd, params)
            
            response_json = json.dumps(response_data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response_json))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self.wfile.write(response_json)
            
        except Exception as e:
            logging.error(f"Failed to handle control request: {e}")
            response_data = {"message": f"Server Error: {e}", "af_finished": False}
            response_json = json.dumps(response_data).encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response_json))
            self.end_headers()
            self.wfile.write(response_json)

    def log_message(self, format, *args):
        """
        覆寫 (Override) 預設的日誌函式，
        以過濾掉 /control?cmd=get_status 的日誌。
        """
        if len(args) > 0 and isinstance(args[0], str) and "GET /control?cmd=get_status" in args[0]:
            return # 安靜地忽略
        
        super().log_message(format, *args)

# --- (StreamingServer 類別不變) ---
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    
    def __init__(self, server_address, RequestHandlerClass, output, light_control, motor_control, camera_manager, app):
        self.output = output
        self.light_control = light_control
        self.motor_control = motor_control
        self.camera_manager = camera_manager
        self.app = app
        
        self.bf_light_state = False
        self.fluo_light_state = False
        
        self.api_handler = ApiHandler(
            camera_manager=self.camera_manager,
            motor_control=self.motor_control,
            light_control=self.light_control,
            server_instance=self 
        )
        
        super().__init__(server_address, RequestHandlerClass)

# --- (__main__ 區塊已修改) ---
if __name__ == "__main__":
    app = QCoreApplication(sys.argv)
    picam2 = Picamera2()
    
    Picamera2.set_logging(logging.INFO)
    
    # 確保資料夾存在
    os.makedirs(config.FOCUS_FOLDER, exist_ok=True)
    os.makedirs(config.CAPTURE_FOLDER, exist_ok=True)
    
    status_cb = lambda msg: logging.info(f"STATUS: {msg}")
    
    motor_control = MotorControl(status_cb)
    light_control = LightControl(bf_pin=config.BRIGHT_FIELD_PIN, fluo_pin=config.FLUORESCENCE_PIN, status_callback=status_cb)
    
    output = StreamingOutput()
    
    camera_manager = CameraManager(picam2, output, motor_control, app)
    
    camera_manager.status_updated.connect(status_cb)

    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler, output, light_control, motor_control, camera_manager, app)
        
        logging.info(f"Starting camera with default settings: {config.DEFAULT_RESOLUTION_KEY} @ {config.DEFAULT_FPS_KEY} FPS")
        camera_manager.restart_camera_system(config.DEFAULT_RESOLUTION_KEY, config.DEFAULT_FPS_KEY)

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        logging.info(f"✅ Web server started. Access at http://'raspberry pi ip']/index.html")
        logging.info(f"✅ Web server started. Access at http://localhost:8000/index.html")

        def shutdown():
            logging.info("Shutting down...")
            if 'server' in locals() and server:
                 server.shutdown()
            if 'server_thread' in locals() and server_thread.is_alive():
                 server_thread.join(timeout=1)
            
            # 1. 只發送 "stop" 事件
            if camera_manager.camera_worker:
                 QMetaObject.invokeMethod(camera_manager.camera_worker, "stop", Qt.QueuedConnection)
            
            if camera_manager.camera_thread: 
                 if not camera_manager.camera_thread.wait(3000):
                    logging.warning("CameraWorker thread did not shut down in time.")
                 else:
                    logging.info("CameraWorker thread shut down successfully.")
            
            if picam2.started:
                logging.info("Ensuring picam2 is stopped.")
                picam2.stop()
            motor_control.close()
            light_control.cleanup()
            logging.info("Shutdown complete.")

        app.aboutToQuit.connect(shutdown)
        sys.exit(app.exec())

    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        if 'server' in locals() and server: server.shutdown()
        if 'picam2' in locals() and picam2.started: picam2.stop()
        if 'motor_control' in locals(): motor_control.close()
        if 'light_control' in locals(): light_control.cleanup()
        sys.exit(1)