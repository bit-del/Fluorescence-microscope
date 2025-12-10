# analyze_stack.py
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
# 移除了 tkinter

# --- 各種對焦演算法 (Metrics) ---

def score_variance(image):
    """1. 【Variance】 標準變異數法 (統計類)"""
    return np.var(image)

def score_sobel_energy(image):
    """2. 【Sobel Energy】 梯度能量 (梯度類)"""
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return np.mean(magnitude ** 2)

def score_laplacian_variance(image):
    """3. 【Laplacian Variance】 二階微分變異數 (二階導數類)"""
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    return np.var(lap)

def score_tenengrad(image):
    """4. 【Tenengrad】 Sobel 梯度平方和 (梯度類)"""
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_sq = gx**2 + gy**2
    return np.mean(gradient_sq)

def score_brenner(image):
    """5. 【Brenner】 相隔兩點差分平方和 (計算快速)"""
    image = image.astype(np.float32)
    diff_x = image[:, 2:] - image[:, :-2]
    return np.mean(diff_x ** 2)

def score_fluo_morph_sobel(image):
    """
    6. 【Fluo Original】 (螢光專用)
    邏輯: Morphological Opening (去鬼影) + Sobel Edge + Top 0.1% Peaks
    """
    img_float = image.astype(np.float32)

    # 去鬼影
    kernel_size = 13 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    structure_only = cv2.morphologyEx(img_float, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(structure_only, (5, 5), 0)

    # Sobel
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    
    # 取前 0.1%
    flat_mag = magnitude.flatten()
    total_pixels = len(flat_mag)
    if total_pixels == 0: return 0.0
    
    top_n_count = int(total_pixels * 0.001) 
    if top_n_count < 10: top_n_count = 10
    
    top_gradients = np.partition(flat_mag, -top_n_count)[-top_n_count:]
    score = np.mean(top_gradients ** 2)

    return score

# --- 主程式邏輯 ---

def analyze_folder():
    # 【修改】使用您指定的路徑 (請依需求自行修改這裡)
    # folder_path = r"/home/pi/Desktop/GUI/pifp_data/Z_Stack_20251209_080223"
    folder_path = r"/home/pi/Desktop/GUI/pifp_data/Z_Stack_20251209_082013"
    
    if not os.path.exists(folder_path):
        print(f"錯誤：找不到資料夾 {folder_path}")
        return

    csv_path = os.path.join(folder_path, "data_log.csv")
    if not os.path.exists(csv_path):
        print("錯誤：找不到 data_log.csv")
        return

    # 讀取 CSV
    data_map = {} 
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get('Filename')
                pos_str = row.get('Position_um')
                if fname and pos_str:
                    data_map[fname] = float(pos_str)
    except Exception as e:
        print(f"讀取失敗: {e}")
        return

    sorted_files = sorted(data_map.keys(), key=lambda k: data_map[k])
    sorted_positions = [data_map[f] for f in sorted_files]

    print(f"開始分析 {len(sorted_files)} 張圖片...")

    # 初始化結果容器
    results = {
        "Variance": [],
        "Sobel Energy": [],
        "Laplacian Var": [],
        "Tenengrad": [],
        "Brenner": [],
        "Fluo (Morph+0.1%)": []
    }

    total_files = len(sorted_files)
    for i, fname in enumerate(sorted_files):
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            for k in results: results[k].append(0)
            continue

        gray = img[:, :, 1] # 使用綠色通道
        
        results["Variance"].append(score_variance(gray))
        results["Sobel Energy"].append(score_sobel_energy(gray))
        results["Laplacian Var"].append(score_laplacian_variance(gray))
        results["Tenengrad"].append(score_tenengrad(gray))
        results["Brenner"].append(score_brenner(gray))
        results["Fluo (Morph+0.1%)"].append(score_fluo_morph_sobel(gray))

        if (i + 1) % 10 == 0:
            print(f"進度: {i + 1}/{total_files}...")

    # 繪圖
    print("正在繪製圖表...")
    plt.figure(figsize=(14, 8))
    
    # 1. 定義線條樣式循環 (Linestyles) - 已移除實線 '-'
    linestyles = ['-']
    
    # 2. 定義數據點符號循環 (Markers)
    markers = ['o', 's', '^', 'x', 'D', '*']
    
    for i, (method_name, scores) in enumerate(results.items()):
        scores_arr = np.array(scores)
        min_v = np.min(scores_arr)
        max_v = np.max(scores_arr)
        
        if max_v - min_v == 0:
            norm_scores = scores_arr
        else:
            norm_scores = (scores_arr - min_v) / (max_v - min_v)
        
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        
        plt.plot(sorted_positions, norm_scores, label=method_name, 
                 linewidth=1.5,      # 線寬
                 linestyle=ls,       # 線的樣式
                 marker=mk,          # 點的樣式
                 markersize=6,       # 點的大小
                 markevery=10,       # 【修改】每 10 個點才畫一個符號，避免擁擠
                 alpha=0.8)          # 透明度

    folder_name = os.path.basename(folder_path)
    plt.title(f"6 Metrics Comparison (Spaced Markers)\nDataset: {folder_name}", fontsize=14)
    plt.xlabel("Z Position (um)", fontsize=12)
    plt.ylabel("Normalized Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=0, color='k', linestyle=':', alpha=0.3)
    
    save_path = os.path.join(folder_path, "analysis_spaced_markers.png")
    plt.savefig(save_path)
    print(f"完成！圖表已存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    analyze_folder()