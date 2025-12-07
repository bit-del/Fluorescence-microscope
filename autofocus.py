# autofocus.py
import time
import logging
import numpy as np
import cv2

# --- 對焦指數計算 (Metrics) ---

def compute_score_fluo(image):
    """
    [螢光專用 - Fluo] 
    邏輯: Morphological Opening (去鬼影) + Sobel Edge Detection
    """
    img_float = image.astype(np.float32)

    # 1. 形態學去鬼影
    kernel_size = 13 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    structure_only = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(structure_only, (5, 5), 0)

    # 2. 計算 Sobel 梯度
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    
    # 3. 取前 0.1% 最亮點
    flat_mag = magnitude.flatten()
    total_pixels = len(flat_mag)
    if total_pixels == 0: return 0.0
    
    top_n_count = int(total_pixels * 0.001) 
    if top_n_count < 10: top_n_count = 10
    
    top_gradients = np.partition(flat_mag, -top_n_count)[-top_n_count:]
    score = np.mean(top_gradients ** 2)

    return score

def compute_score_bf(image):
    """
    [明視野專用 - BF]
    邏輯: Global Variance (全域變異數)
    原因: 明視野正焦時黑色最深（對比度最高），變異數最大。
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算全域變異數
    score = np.var(image)
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
        logging.info("FocusAlgorithm initialized.")
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
        
        # 反轉方向：利用這一點，如果在上一步結束時我們剛好在峰值旁邊（Back 1 step），
        # 反轉後就會直接朝峰值移動。
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
                
                # --- 情況 A: 剛開始就走錯方向 (Step 1 專用) ---
                if (check_threshold and self.current_step_index == 1 and self.drop_count == 1 and not self.wrong_way_check_done): 
                    self.wrong_way_check_done = True 
                    self.direction *= -1 
                    step = step_size * self.direction * 1.5 
                    command = f"z{step}"
                    self.last_value = current_value
                    self.maximum_value = current_value
                    self.steps_from_peak = 0 
                    self.drop_count = 0
                    self.status_updater(f"AF: Wrong direction. Reversing...")
                    return {"type": "MOVE", "command": command}

                # --- 情況 B: 分數暴跌 (>30%) ---
                elif (check_threshold and self.maximum_value > 0 and (current_value / self.maximum_value) < 0.7):
                    if is_last_step:
                        # 最後一步：必須精確回到峰值
                        back_steps = self.steps_from_peak  
                        step = step_size * self.direction * -1 * back_steps
                        msg = f"AF: Final Step Drop (70%). Back to Peak ({back_steps} steps)."
                    else:
                        # 中間步驟：只要退回上一步 (Back 1 step)
                        step = step_size * self.direction * -1
                        msg = f"AF: Peak missed (70% Drop). Back 1 step to refine."

                    command = f"z{step}"
                    self.status_updater(msg)
                    self.state = self.STATE_STEPPING_BACK 
                    return {"type": "MOVE", "command": command}

                # --- 情況 C: 連續下降兩次 ---
                elif self.drop_count >= 2:
                    if is_last_step:
                        # 最後一步：必須精確回到峰值
                        back_steps = self.steps_from_peak
                        step = step_size * self.direction * -1 * back_steps
                        msg = f"AF: Final Step (2 drops). Back to Peak ({back_steps} steps)."
                    else:
                        # 中間步驟：只要退回上一步 (Back 1 step)
                        step = step_size * self.direction * -1
                        msg = f"AF: Peak passed (2 drops). Back 1 step to refine."
                    
                    command = f"z{step}"
                    self.status_updater(msg)
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