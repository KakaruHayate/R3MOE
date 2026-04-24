import os
import json
import re
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from datetime import datetime
import argparse

def parse_session_name(session_str):
    date_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', session_str)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%Y-%m-%d_%H-%M-%S")
            return (1, dt.timestamp())
        except:
            pass
    range_match = re.search(r'(\d+)_(\d+)-(\d+)_(\d+)', session_str)
    if range_match:
        return (2, float(f"{range_match.group(1)}.{range_match.group(2)}"))
    return (3, session_str)

def process_error_value(timestamps, values):
    """
    检测数据开头可能存在的恒定无效值段。
    从第二个样本开始寻找第一个不同于初始值的样本，
    若变化发生在第二个样本（即开头仅一个点），则认为不是错误段（返回空）。
    若全程无变化，则整个序列为错误段。
    """
    if len(values) < 2:
        return []  # 数据不足，不判定错误段

    first_val = values[0]
    start_time = timestamps[0]

    # 寻找第一个不同于初始值的索引
    change_idx = -1
    for i in range(1, len(values)):
        if values[i] != first_val:
            change_idx = i
            break

    if change_idx == -1:
        # 全程无变化，全部为错误段
        return [(start_time, timestamps[-1])]
    elif change_idx == 1:
        # 只在第二个点就发生了变化，说明开头只有单个样本，视为正常波动
        return []
    else:
        # 存在一段恒定的无效值
        return [(start_time, timestamps[change_idx])]

def extract_b_worker(args):
    csv_path, root_path, fps, max_b = args
    try:
        df = pd.read_csv(csv_path, usecols=['TimeStamp', 'jawOpen', 'mouthClose'], engine='c')
        t, x_raw = df['TimeStamp'].values, df['jawOpen'].values - df['mouthClose'].values
        
        if len(t) < 10:
            return {"path": str(csv_path), "status": "skip"}

        error_segments = process_error_value(t, x_raw)
        duration = float(t[-1] - t[0])

        t_uniform = np.arange(t[0], t[-1], 1.0 / fps)
        x = np.interp(t_uniform, t, x_raw)
        
        valid_mask = np.ones_like(t_uniform, dtype=bool)
        for e_start, e_end in error_segments:
            valid_mask &= ~((t_uniform >= e_start) & (t_uniform <= e_end))
        x_clean = x[valid_mask]
        
        if len(x_clean) < 10:
            x_clean = x 

        x_min, x_max = x_clean.min(), x_clean.max()
        bins = np.linspace(x_min - 0.05, x_max + 0.05, 200)
        counts, edges = np.histogram(x_clean, bins=bins)
        
        smoothed = gaussian_filter1d(counts, sigma=1.2, mode='constant', cval=0.0)
        max_density = np.max(smoothed)
        
        peaks, _ = find_peaks(smoothed, height=max_density * 0.02, prominence=max_density * 0.01)
        
        if len(peaks) > 0:
            raw_b = round(float(edges[peaks[0]] + (edges[1] - edges[0]) / 2.0), 5)
        else:
            raw_b = round(float(edges[np.argmax(smoothed)] + (edges[1] - edges[0]) / 2.0), 5)

        rel_path = Path(csv_path).relative_to(root_path)
        spk_name = rel_path.parts[0] if len(rel_path.parts) > 1 else "Unknown_SPK"

        return {
            "path": str(csv_path),
            "rel_path": str(rel_path),
            "spk": spk_name,
            "session": Path(csv_path).parent.name,
            "sort_key": parse_session_name(Path(csv_path).parent.name),
            "raw_b": raw_b,
            "is_purple": raw_b > max_b,  
            "duration": duration,
            "error_segments": error_segments
        }
    except: 
        return None

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}小时 {m}分钟 {s:.1f}秒"

def main():
    parser = argparse.ArgumentParser(description='数据清洗：提取b值，丢弃异常片段，生成b_values.json')
    parser.add_argument('--root_raw', required=True, help='原始数据根目录')
    parser.add_argument('--root_waste', default='./waste', help='废弃数据回收目录（默认 ./waste）')
    parser.add_argument('--json_output', default='b_values.json', help='输出的参数文件名（存放在root_raw下）')
    parser.add_argument('--fps', type=float, default=50.0, help='采样帧率')
    parser.add_argument('--max_b', type=float, default=0.06, help='紫色阈值，超过即整段废弃')
    args = parser.parse_args()

    root_raw = Path(args.root_raw)
    root_waste = Path(args.root_waste)
    json_output_name = args.json_output
    fps = args.fps
    max_b = args.max_b

    if not root_raw.exists():
        print(f"Error: Source directory {root_raw} not found.")
        return

    csv_files = list(root_raw.rglob('mouth_data.csv'))
    print(f"Found {len(csv_files)} CSV files. Starting parallel analysis...")

    task_args = [(f, root_raw, fps, max_b) for f in csv_files]

    spk_dict = {}
    waste_folders = set()
    total_wasted_sec = 0.0
    total_kept_sec = 0.0
    final_b_map = {}

    num_processes = min(cpu_count(), 8)
    with Pool(processes=num_processes, maxtasksperchild=20) as pool:
        for r in pool.imap_unordered(extract_b_worker, task_args, chunksize=10):
            if r is None or r.get('status') == 'skip':
                continue
            spk_dict.setdefault(r['spk'], []).append(r)

    print("Executing Band clamping and waste evaluation...")
    
    for spk, items in spk_dict.items():
        items_sorted = sorted(items, key=lambda x: (x['sort_key'][0], x['sort_key'][1]))
        
        valid_vals = np.array([x['raw_b'] for x in items_sorted if not x['is_purple']])
        if len(valid_vals) >= 4:
            q1, q3 = np.percentile(valid_vals, [25, 75])
            iqr = max(q3 - q1, 0.002) 
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            low, high = -0.01, max_b

        for item in items_sorted:
            if item['is_purple']:
                waste_folders.add(Path(item['path']).parent)
                total_wasted_sec += item['duration']
            else:
                clamped_b = float(np.clip(item['raw_b'], low, high))
                final_b_map[item['rel_path']] = {
                    "b_val": clamped_b,
                    "error_segments": item['error_segments']
                }
                total_kept_sec += item['duration']

    print(f"\nMoving waste data to {root_waste}...")
    moved_count = 0
    for folder in waste_folders:
        rel_folder = folder.relative_to(root_raw)
        target_path = root_waste / rel_folder
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if target_path.exists(): 
                shutil.rmtree(target_path) 
            shutil.move(str(folder), str(target_path))
            moved_count += 1
        except Exception as e:
            print(f"  [Warning] Failed to move {folder}: {e}")

    json_path = root_raw / json_output_name
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_b_map, f, indent=2, ensure_ascii=False)

    print(f"\n================ Report ================")
    print(f"Kept records: {len(final_b_map)}")
    print(f"Kept duration: {format_duration(total_kept_sec)}")
    print(f"----------------------------------------")
    print(f"Wasted folders: {moved_count}")
    print(f"Wasted duration: {format_duration(total_wasted_sec)}")
    print(f"----------------------------------------")
    print(f"JSON saved to: {json_path}")
    print(f"========================================")

if __name__ == "__main__":
    main()