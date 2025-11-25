# autofocus.py
import time
import logging
import numpy as np
import os
import glob
import cv2
import config

# --- 對焦指數計算 ---

def compute_variance_function(image):
    """計算影像的變異數 (Variance) 對焦指數"""
    mean_value = np.mean(image)
    variance = np.sum((image - mean_value) ** 2)
    return variance

def compute_brenner_function(image):
    """計算影像的 Brenner 對焦指數"""
    image = np.asarray(image, dtype=np.float32)
    shifted_image = np.roll(image, -2, axis=0)
    difference = shifted_image - image
    difference_squared = np.square(difference)
    brenner_value = np.sum(difference_squared[:-2, :])
    return brenner_value

def compute_laplacian_function(image):
    """
    [螢光專用] 版本 V6 (去除鬼影版): Morphological Opening + Sobel
    """
    # 1. 轉為浮點數
    img_float = image.astype(np.float32)

    # --- 關鍵步驟 A: 形態學去鬼影 ---
    kernel_size = 13 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    structure_only = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(structure_only, (5, 5), 0)

    # --- 關鍵步驟 B: 計算結構邊緣 ---
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    
    # --- 步驟 C: 評分 ---
    flat_mag = magnitude.flatten()
    total_pixels = len(flat_mag)
    if total_pixels == 0: return 0.0
    
    top_n_count = int(total_pixels * 0.001) # 0.1%
    if top_n_count < 10: top_n_count = 10
    top_gradients = np.partition(flat_mag, -top_n_count)[-top_n_count:]
    score = np.mean(top_gradients ** 2)

    return score

# 【新增】明視野專用
def compute_variance_of_laplacian(image):
    """
    [明視野專用 - 改良版] Laplacian 變異數 + 抗噪
    加入 Gaussian Blur (5x5) 以濾除高頻雜訊，避免失焦時雜訊分數過高的問題。
    """
    # 1. 高斯模糊 (關鍵步驟：抹除 Image 008 那種細碎雜訊)
    # kernel=(5,5) 是顯微鏡常用的參數，能保留結構但殺死雜訊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. 計算 Laplacian (偵測邊緣)
    # 使用 64-bit float 避免溢位
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # 3. 計算變異數 (Variance) 作為分數
    score = laplacian.var()
    
    return score

class FocusAlgorithm:
    STATE_IDLE = 0
    STATE_START_STEP = 1
    STATE_SEARCHING = 2
    STATE_STEPPING_BACK = 3 
    STATE_FINISHING = 4

    def __init__(self, steps_config, status_updater):
        self.steps_config = steps_config
        self.status_updater = status_updater
        logging.info("Non-blocking FocusAlgorithm initialized.")
        self.reset()

    def reset(self):
        self.state = self.STATE_IDLE
        self.current_step_index = 0
        self.current_step_config = None
        self.direction = -1
        self.last_value = 0
        self.maximum_value = 0
        self.drop_count = 0
        self.steps_from_peak = 0 
        self.wrong_way_check_done = False 

    def start(self):
        if self.state != self.STATE_IDLE:
            logging.warning("AF Algorithm: Start called when not idle.")
        self.reset()
        self.state = self.STATE_START_STEP
        self.status_updater("Continuous AF started...")

    def cancel(self):
        self.state = self.STATE_IDLE
        self.status_updater("Continuous AF cancelled.")

    def _load_next_step(self):
        if self.current_step_index >= len(self.steps_config):
            self.state = self.STATE_FINISHING
            return False
            
        self.current_step_config = self.steps_config[self.current_step_index]
        self.current_step_index += 1
        
        self.direction *= -1
        self.last_value = 0
        self.maximum_value = -1 
        self.drop_count = 0
        self.steps_from_peak = 0 
        
        step_size, method, _ = self.current_step_config
        self.status_updater(f"AF: Starting step {self.current_step_index} (Size: {step_size}, Method: {method})")
        return True

    def step(self, current_value):
        if self.state == self.STATE_IDLE:
            return {"type": "WAIT"}

        if self.state == self.STATE_FINISHING:
            self.state = self.STATE_IDLE
            return {"type": "FINISHED"}

        if self.state == self.STATE_START_STEP:
            if not self._load_next_step():
                return {"type": "FINISHED"}
            self.last_value = current_value
            self.maximum_value = current_value
            self.steps_from_peak = 0 
            self.state = self.STATE_SEARCHING
            step_size, _, _ = self.current_step_config
            command = f"z{step_size * self.direction}" 
            return {"type": "MOVE", "command": command}

        if self.state == self.STATE_STEPPING_BACK:
            self.state = self.STATE_START_STEP
            return {"type": "WAIT"}

        if self.state == self.STATE_SEARCHING:
            step_size, _, check_threshold = self.current_step_config
            is_last_step = (self.current_step_index >= len(self.steps_config))
            
            if current_value > self.maximum_value:
                self.maximum_value = current_value
                self.steps_from_peak = 0 
            else:
                self.steps_from_peak += 1 
            
            if current_value < self.last_value:
                self.drop_count += 1
                if (check_threshold and self.current_step_index == 1 and self.drop_count == 1 and not self.wrong_way_check_done): 
                    self.wrong_way_check_done = True 
                    self.direction *= -1 
                    step = step_size * self.direction * 1.5 
                    command = f"z{step}"
                    self.last_value = current_value
                    self.maximum_value = current_value
                    self.steps_from_peak = 0 
                    self.drop_count = 0
                    return {"type": "MOVE", "command": command}

                elif (check_threshold and self.maximum_value > 0 and current_value / self.maximum_value < 0.7):
                    if is_last_step:
                        back_steps = self.steps_from_peak  
                        step = step_size * self.direction * -1 * back_steps
                        command = f"z{step}"
                        self.status_updater(f"AF: Final Step Drop (70%). Returning to Peak ({back_steps} steps): {command}")
                        self.state = self.STATE_STEPPING_BACK
                        return {"type": "MOVE", "command": command}
                    else:
                        self.status_updater(f"AF: Peak missed (below 70%). Advancing to next step.")
                        self.state = self.STATE_START_STEP 
                        return {"type": "WAIT"}

                elif self.drop_count >= 2:
                    if is_last_step:
                        back_steps = self.steps_from_peak + 1 
                        step = step_size * self.direction * -1 * back_steps
                        command = f"z{step}"
                        self.status_updater(f"AF: Final Step Finished. Returning {back_steps} steps to Peak: {command}")
                        self.state = self.STATE_STEPPING_BACK 
                        return {"type": "MOVE", "command": command}
                    else:
                        step = step_size * self.direction * -1
                        command = f"z{step}"
                        self.status_updater(f"AF: Peak missed (2 drops). Stepping back {command}, then advancing.")
                        self.state = self.STATE_STEPPING_BACK 
                        return {"type": "MOVE", "command": command}
                
                else:
                    step = step_size * self.direction * (1.5 if check_threshold else 1.0)
                    self.last_value = current_value
                    command = f"z{step}"
                    return {"type": "MOVE", "command": command}
            else:
                self.drop_count = 0
                self.last_value = current_value
                command = f"z{step_size * self.direction}"
                return {"type": "MOVE", "command": command}
        return {"type": "WAIT"}