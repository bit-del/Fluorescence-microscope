# image_processing.py
import numpy as np
import cv2
import logging
from PySide6.QtGui import QImage

# 基礎黑電平
DEFAULT_SENSOR_BLACK = 64.0
FLUORESCENCE_NOISE_THRESHOLD = 2.0
FLUORESCENCE_DIGITAL_GAIN = 1.5 

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
    使用「空間變異矩陣」進行顏色矯正
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

    # 執行每個像素的矩陣乘法: Corrected = Original @ Matrix.T
    # 展開運算以提升效能
    m00 = unmix_tensor[:, :, 0, 0]; m01 = unmix_tensor[:, :, 0, 1]; m02 = unmix_tensor[:, :, 0, 2]
    m10 = unmix_tensor[:, :, 1, 0]; m11 = unmix_tensor[:, :, 1, 1]; m12 = unmix_tensor[:, :, 1, 2]
    m20 = unmix_tensor[:, :, 2, 0]; m21 = unmix_tensor[:, :, 2, 1]; m22 = unmix_tensor[:, :, 2, 2]

    r_out = m00 * r + m01 * g + m02 * b
    g_out = m10 * r + m11 * g + m12 * b
    b_out = m20 * r + m21 * g + m22 * b
    
    return r_out, g_out, b_out

def apply_correction(science_image_raw: tuple, 
                     gain_maps: tuple, 
                     bg_subtract_enabled: bool, 
                     bg_frame_raw: tuple,
                     unmix_matrix: np.ndarray = None) -> np.ndarray:
    """
    整合校正流程 (修正順序版)
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
        sample_pixels = g_curr[::10, ::10] 
        current_floor = np.percentile(sample_pixels, 1) 
        
        if current_floor > 0:
            r_curr -= current_floor
            g_curr -= current_floor
            b_curr -= current_floor
        
        # 去負值
        r_curr = np.maximum(r_curr, 0)
        g_curr = np.maximum(g_curr, 0)
        b_curr = np.maximum(b_curr, 0)

    # Step 3: 應用平場校正 (Flat Field / Gain Maps)
    # 先把亮度修平、暗角修掉，還原成線性的物理光強
    if gain_maps is not None:
        gm_r, gm_g, gm_b = gain_maps
        
        if not bg_subtract_enabled:
            # 明視野模式：各通道獨立校正
            r_out = r_curr * gm_r
            g_out = g_curr * gm_g
            b_out = b_curr * gm_b
        else:
            # 螢光模式：使用幾何校正 (Geometry Mode)
            # 統一使用 Green Gain Map，修正幾何光學暗角，不改變顏色比例
            r_out = r_curr * gm_g
            g_out = g_curr * gm_g
            b_out = b_curr * gm_g
    else:
        r_out, g_out, b_out = r_curr, g_curr, b_curr

    # Step 4: 應用顏色校正 (Unmix) <-- 移到這裡！
    # 現在輸入的數據已經是亮度均勻的了，Unmix 矩陣可以正確地分離顏色
    if unmix_matrix is not None:
        r_out, g_out, b_out = apply_color_unmix(r_out, g_out, b_out, unmix_matrix)
        
        # Unmix 運算後，某些極端顏色可能會變成負值，需歸零
        r_out = np.maximum(r_out, 0)
        g_out = np.maximum(g_out, 0)
        b_out = np.maximum(b_out, 0)

    # Step 5: 數位增亮 (Digital Gain)
    if bg_subtract_enabled:
        r_out *= FLUORESCENCE_DIGITAL_GAIN
        g_out *= FLUORESCENCE_DIGITAL_GAIN
        b_out *= FLUORESCENCE_DIGITAL_GAIN

    # Step 6: 最終輸出 (Clip)
    np.clip(r_out, 0, 65535, out=r_out)
    np.clip(g_out, 0, 65535, out=g_out)
    np.clip(b_out, 0, 65535, out=b_out)
    
    corrected_rgb = np.stack([r_out, g_out, b_out], axis=-1)
    
    # Step 7: 去噪 (Denoise)
    if bg_subtract_enabled:
        corrected_rgb = corrected_rgb.astype(np.uint16)
        corrected_rgb = cv2.medianBlur(corrected_rgb, 3)
        
    return corrected_rgb.astype(np.uint16)

def convert_to_qimage(rgb_16bit_array: np.ndarray, ev_comp: float) -> QImage:
    """ 轉為 8-bit 顯示 """
    max_val = 65535.0
    if max_val > 0:
        normalized = rgb_16bit_array.astype(np.float32) / max_val
        normalized *= ev_comp
        np.clip(normalized, 0, 1, out=normalized)
        preview_arr_8bit = (normalized * 255).astype(np.uint8)
        h, w, c = preview_arr_8bit.shape
        q_img = QImage(preview_arr_8bit.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return q_img.copy()
    return QImage()