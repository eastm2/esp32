# build_dataset_from_csv.py —— 从分传感器 CSV 生成标定窗口数据集
import json
import numpy as np
from collections import deque
from pathlib import Path
import pandas as pd

# ================== 配置 ==================
FS = 50
WINDOW_SIZE = 128
OVERLAP = 0.5
STEP = int(WINDOW_SIZE * (1 - OVERLAP))   # 64

ACC_SCALE = 1 / 16384.0      # ±2g
GYRO_SCALE = 1 / 131.0       # ±250°/s
B_NORM = 46.83               # 地磁场强度 μT

INPUT_DIR = Path("collected_raw_data")        # CSV 文件所在目录
OUTPUT_NPZ = Path("dataset_from_csv.npz")     # 输出 npz 文件
CALIB_JSON = Path("calib_params.json")        # 标定文件

# ================== 加载标定参数 ==================
with open(CALIB_JSON, 'r') as f:
    calib = json.load(f)

M_acc = np.array(calib["MPU6050_accelerometer"]["M_matrix"])
b_acc = np.array(calib["MPU6050_accelerometer"]["bias_vector"])
b_gyro = np.array(calib["MPU6050_gyroscope"]["bias_vector"])

mag_params = calib["HMC5883L_magnetometer"]
A_lsb = np.array(mag_params["transform_matrix_A_half"])
c_lsb = np.array(mag_params["center_offset_c"])

# ================== 标定函数 ==================
def calibrate_acc(ax_raw, ay_raw, az_raw):
    ax = ax_raw * ACC_SCALE
    ay = ay_raw * ACC_SCALE
    az = az_raw * ACC_SCALE
    a_raw = np.array([ax - b_acc[0], ay - b_acc[1], az - b_acc[2]])
    a_cal = M_acc @ a_raw
    return a_cal[0], a_cal[1], a_cal[2]

def calibrate_gyro(gx_raw, gy_raw, gz_raw):
    gx = gx_raw * GYRO_SCALE
    gy = gy_raw * GYRO_SCALE
    gz = gz_raw * GYRO_SCALE
    return gx - b_gyro[0], gy - b_gyro[1], gz - b_gyro[2]

def calibrate_mag(mx_raw, my_raw, mz_raw):
    mag_lsb = np.array([mx_raw, my_raw, mz_raw])
    centered = mag_lsb - c_lsb
    mag_norm = A_lsb @ centered
    return (mag_norm * B_NORM).tolist()

# ================== 滑动窗口缓冲 ==================
class SlidingWindowBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=WINDOW_SIZE)
        self.last_output_time = -STEP / FS

    def add_point(self, ax, ay, az, gx, gy, gz, mx, my, mz):
        self.buffer.append([ax, ay, az, gx, gy, gz, mx, my, mz])

    def ready(self, current_time):
        return len(self.buffer) >= WINDOW_SIZE and (current_time - self.last_output_time) >= (STEP / FS)

    def get_window(self):
        self.last_output_time += STEP / FS
        return np.array(list(self.buffer))

# ================== 主处理流程 ==================
def process_pair(acc_file, mag_file, subject_id, label):
    """读取一对 CSV 文件，标定并滑动窗口，返回窗口数组 (N,128,9)"""
    df_acc = pd.read_csv(acc_file)
    df_mag = pd.read_csv(mag_file)
    # 对齐长度
    min_len = min(len(df_acc), len(df_mag))
    df_acc = df_acc.iloc[:min_len]
    df_mag = df_mag.iloc[:min_len]

    # 原始 ADC 列
    ax_raw = df_acc["acc_x_raw"].values
    ay_raw = df_acc["acc_y_raw"].values
    az_raw = df_acc["acc_z_raw"].values
    gx_raw = df_acc["gyro_x_raw"].values
    gy_raw = df_acc["gyro_y_raw"].values
    gz_raw = df_acc["gyro_z_raw"].values

    mx_raw = df_mag["mag_x_raw"].values
    my_raw = df_mag["mag_y_raw"].values
    mz_raw = df_mag["mag_z_raw"].values

    # 时间戳（毫秒转秒）
    ts_sec = df_acc["timestamp_ms"].values / 1000.0

    buf = SlidingWindowBuffer()
    windows = []
    for i in range(min_len):
        ax, ay, az = calibrate_acc(ax_raw[i], ay_raw[i], az_raw[i])
        gx, gy, gz = calibrate_gyro(gx_raw[i], gy_raw[i], gz_raw[i])
        mx, my, mz = calibrate_mag(mx_raw[i], my_raw[i], mz_raw[i])
        buf.add_point(ax, ay, az, gx, gy, gz, mx, my, mz)

        t = ts_sec[i]
        if buf.ready(t):
            win = buf.get_window()
            windows.append(win)

    if not windows:
        return None
    return np.stack(windows, axis=0)   # (N,128,9)

def main():
    # 查找所有 acc_gyro 文件，并匹配对应的 mag 文件
    acc_files = sorted(INPUT_DIR.glob("sub*_label*_acc_gyro*.csv"))
    all_windows = []
    all_labels = []
    all_subjects = []
    all_win_idx = []
    global_idx = 0

    for acc_f in acc_files:
        # 文件名示例：sub1_label0_acc_gyro_20240624_123456.csv
        name = acc_f.stem
        parts = name.split('_')
        subj = int(parts[0][3:])
        lab = int(parts[1][5:])
        # 寻找对应的 mag 文件
        mag_files = list(INPUT_DIR.glob(f"sub{subj}_label{lab}_mag*.csv"))
        if not mag_files:
            print(f"跳过被试{subj}标签{lab}：缺少磁力计文件")
            continue
        mag_f = sorted(mag_files)[-1]  # 取最新

        wins = process_pair(acc_f, mag_f, subj, lab)
        if wins is None:
            continue
        n_win = len(wins)
        # 拆分为 acc, gyro, mag (128,3)
        acc_w = wins[:, :, :3]
        gyro_w = wins[:, :, 3:6]
        mag_w = wins[:, :, 6:9]

        all_windows.append((acc_w, gyro_w, mag_w))
        all_labels.append(np.full(n_win, lab, dtype=np.int32))
        all_subjects.append(np.full(n_win, subj, dtype=np.int32))
        all_win_idx.append(np.arange(global_idx, global_idx + n_win, dtype=np.int32))
        global_idx += n_win
        print(f"被试{subj}活动{lab}: {n_win} 窗口")

    if not all_windows:
        print("未生成任何窗口")
        return

    # 合并
    acc_all = np.concatenate([w[0] for w in all_windows], axis=0)
    gyro_all = np.concatenate([w[1] for w in all_windows], axis=0)
    mag_all = np.concatenate([w[2] for w in all_windows], axis=0)
    labels_arr = np.concatenate(all_labels)
    subjects_arr = np.concatenate(all_subjects)
    win_idx_arr = np.concatenate(all_win_idx)
    has_mag_arr = np.ones(len(acc_all), dtype=bool)

    np.savez_compressed(
        OUTPUT_NPZ,
        acc=acc_all,
        gyro=gyro_all,
        mag=mag_all,
        labels=labels_arr,
        subject_ids=subjects_arr,
        window_indices=win_idx_arr,
        has_mag=has_mag_arr
    )
    print(f"✅ 保存 {len(acc_all)} 个窗口至 {OUTPUT_NPZ}")

if __name__ == "__main__":
    main()