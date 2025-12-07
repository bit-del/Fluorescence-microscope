# image_processing.py
import numpy as np
import cv2
import logging
from PySide6.QtGui import QImage

# 基礎黑電平
DEFAULT_SENSOR_BLACK = 64.0
FLUORESCENCE_NOISE_THRESHOLD = 30
FLUORESCENCE_DIGITAL_GAIN = 2

def get_raw_channels(buffer, raw_config) -> tuple:
    """ 從 Bayer Raw 資料中提取 R, G, B 通道 (Float32) """
    width, height = raw_config["size"]
    stride = raw_config["stride"]
    raw_array_u16 = buffer.view(np.uint16).reshape((height, stride // 2))
    raw_array_trimmed = raw_array_u16[:, :width]
    bayer_float = raw_array_trimmed.astype(np.float32)
    
    blue = bayer_float[0::2, 0::2]
    green1 = bayer_float[0::2, 1::2]
    green2 = bayer_float[1::2, 0::2]
    red = bayer_float[1::2, 1::2]
    green = (green1 + green2) / 2.0
    
    return red, green, blue

def calculate_gain_maps(flat_field_raw: tuple):
    """ 計算 RGB 獨立 Gain Maps """
    logging.info("--- Starting RGB Flat Field Calculation ---")
    r_flat, g_flat, b_flat = flat_field_raw
    
    avg_brightness = np.mean(g_flat)
    if avg_brightness < 200:
        logging.warning("⚠️ Flat Field too dark! Skipping.")
        return None

    h, w = g_flat.shape
    cy, cx = h // 2, w // 2
    center_region = g_flat[cy-50:cy+50, cx-50:cx+50]
    target_brightness = np.percentile(center_region, 95)
    
    # 計算 Gain
    gain_map_r = (target_brightness / np.maximum(r_flat - DEFAULT_SENSOR_BLACK, 1.0)).astype(np.float32)
    gain_map_g = (target_brightness / np.maximum(g_flat - DEFAULT_SENSOR_BLACK, 1.0)).astype(np.float32)
    gain_map_b = (target_brightness / np.maximum(b_flat - DEFAULT_SENSOR_BLACK, 1.0)).astype(np.float32)

    max_gain = 20.0 
    np.clip(gain_map_r, 0, max_gain, out=gain_map_r)
    np.clip(gain_map_g, 0, max_gain, out=gain_map_g)
    np.clip(gain_map_b, 0, max_gain, out=gain_map_b)

    return gain_map_r, gain_map_g, gain_map_b

def apply_color_unmix(r, g, b, unmix_tensor):
    """
    使用 numpy einsum 加速空間變異矩陣運算
    :param r, g, b: 輸入通道 (H, W)
    :param unmix_tensor: 矯正張量 (H, W, 3, 3)
    """
    if unmix_tensor is None:
        return r, g, b
    
    h, w = r.shape
    th, tw = unmix_tensor.shape[:2]
    
    # 確保 Tensor 尺寸與影像匹配
    if th != h or tw != w:
        return r, g, b

    # 優化：使用 Einstein Summation 進行高效張量運算
    # rgb_stack shape: (H, W, 3)
    # unmix_tensor shape: (H, W, 3, 3) -> (H, W, Out_Channel, In_Channel)
    # 運算邏輯: Output[h,w,i] = sum(Unmix[h,w,i,j] * Input[h,w,j])
    rgb_stack = np.stack([r, g, b], axis=-1)
    
    # 'hwij,hwj->hwi'
    # hw: height, width (廣播)
    # i: output channel (R, G, B)
    # j: input channel (R, G, B)
    rgb_out = np.einsum('hwij,hwj->hwi', unmix_tensor, rgb_stack)
    
    return rgb_out[..., 0], rgb_out[..., 1], rgb_out[..., 2]

def apply_correction(science_image_raw: tuple, 
                     gain_maps: tuple, 
                     bg_subtract_enabled: bool, 
                     bg_frame_raw: tuple,
                     unmix_matrix: np.ndarray = None) -> np.ndarray:
    """
    整合校正流程 (修正順序與效能優化版)
    """
    r_in, g_in, b_in = science_image_raw
    
    # Step 1: 基礎扣除 (硬體黑電平)
    r_curr = r_in - DEFAULT_SENSOR_BLACK
    g_curr = g_in - DEFAULT_SENSOR_BLACK
    b_curr = b_in - DEFAULT_SENSOR_BLACK

    # Step 2: 背景扣除 & 動態熱噪 (Background Subtraction)
    if bg_subtract_enabled:
        # A. 扣除固定背景 (漏光)
        if bg_frame_raw is not None:
            bg_r, bg_g, bg_b = bg_frame_raw
            r_curr -= (bg_r - DEFAULT_SENSOR_BLACK)
            g_curr -= (bg_g - DEFAULT_SENSOR_BLACK)
            b_curr -= (bg_b - DEFAULT_SENSOR_BLACK)
        
        # B. 動態熱噪歸零 (Auto-Zeroing)
        # 優化：減少全圖掃描，只取樣
        sample_pixels = g_curr[::10, ::10] 
        current_floor = np.percentile(sample_pixels, 1) 
        
        if current_floor > 0:
            r_curr -= current_floor
            g_curr -= current_floor
            b_curr -= current_floor
        
        # 這裡先不急著 clip 到 0，留到最後或者運算中自動處理，減少指令數
        # 但為了保證 Gain Map 乘法不出現詭異負值增益，這邊做一次 maximum
        r_curr = np.maximum(r_curr, 0)
        g_curr = np.maximum(g_curr, 0)
        b_curr = np.maximum(b_curr, 0)

    # Step 3: 應用平場校正 (Flat Field / Gain Maps)
    if gain_maps is not None:
        gm_r, gm_g, gm_b = gain_maps
        
        if not bg_subtract_enabled:
            # 明視野模式
            r_out = r_curr * gm_r
            g_out = g_curr * gm_g
            b_out = b_curr * gm_b
        else:
            # 螢光模式 (幾何校正)
            r_out = r_curr * gm_g
            g_out = g_curr * gm_g
            b_out = b_curr * gm_g
    else:
        r_out, g_out, b_out = r_curr, g_curr, b_curr

    # Step 4: 應用顏色校正 (Unmix)
    if unmix_matrix is not None:
        r_out, g_out, b_out = apply_color_unmix(r_out, g_out, b_out, unmix_matrix)
        
        # Unmix 後可能產生負值，需歸零
        r_out = np.maximum(r_out, 0)
        g_out = np.maximum(g_out, 0)
        b_out = np.maximum(b_out, 0)

    # Step 5: 數位增亮 (Digital Gain)
    if bg_subtract_enabled:
        r_out *= FLUORESCENCE_DIGITAL_GAIN
        g_out *= FLUORESCENCE_DIGITAL_GAIN
        b_out *= FLUORESCENCE_DIGITAL_GAIN

    # Step 6: 最終輸出 (Clip)
    # 合併 Clip 與 Stack
    np.clip(r_out, 0, 65535, out=r_out)
    np.clip(g_out, 0, 65535, out=g_out)
    np.clip(b_out, 0, 65535, out=b_out)
    
    corrected_rgb = np.stack([r_out, g_out, b_out], axis=-1).astype(np.uint16)
    
    # Step 7: 去噪 (Denoise)
    if bg_subtract_enabled:
        corrected_rgb = cv2.medianBlur(corrected_rgb, 3)
        
    return corrected_rgb

def convert_to_qimage(rgb_16bit_array: np.ndarray, ev_comp: float) -> QImage:
    """ 轉為 8-bit 顯示 """
    max_val = 65535.0
    # 優化：避免全圖除法，改用乘數
    scale_factor = ev_comp / max_val
    
    normalized = rgb_16bit_array.astype(np.float32)
    normalized *= scale_factor
    np.clip(normalized, 0, 1, out=normalized)
    
    preview_arr_8bit = (normalized * 255).astype(np.uint8)
    h, w, c = preview_arr_8bit.shape
    q_img = QImage(preview_arr_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return q_img.copy()