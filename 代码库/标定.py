# calibration.py —— 独立的标定模块
import json
import numpy as np
from pathlib import Path

# ================== 常量 ==================
ACC_SCALE = 1 / 16384.0      # ±2g 量程：LSB 转 g
GYRO_SCALE = 1 / 131.0       # ±250°/s 量程：LSB 转 °/s
B_NORM = 46.83               # 地磁场强度 μT（从训练数据估算）

# ================== 加载标定参数 ==================
def _load_calib(calib_json_path="calib_params.json"):
    """加载标定 JSON 文件，返回标定矩阵和偏置向量"""
    with open(calib_json_path, 'r') as f:
        calib = json.load(f)

    # 加速度计
    M_acc = np.array(calib["MPU6050_accelerometer"]["M_matrix"])
    b_acc = np.array(calib["MPU6050_accelerometer"]["bias_vector"])

    # 陀螺仪
    b_gyro = np.array(calib["MPU6050_gyroscope"]["bias_vector"])

    # 磁力计（LSB 尺度）
    mag_params = calib["HMC5883L_magnetometer"]
    A_lsb = np.array(mag_params["transform_matrix_A_half"])
    c_lsb = np.array(mag_params["center_offset_c"])

    return M_acc, b_acc, b_gyro, A_lsb, c_lsb

# 模块加载时一次性读取参数（若 JSON 文件不存在会报错，可在使用时再调用）
_M_acc, _b_acc, _b_gyro, _A_lsb, _c_lsb = _load_calib()

# ================== 标定函数 ==================
def calibrate_acc(ax_raw, ay_raw, az_raw):
    """
    加速度计原始 ADC → 标定后 g 值
    输入：单点原始 ADC 值（ax_raw, ay_raw, az_raw）
    返回：(ax_cal, ay_cal, az_cal) 单位 g
    """
    ax = ax_raw * ACC_SCALE
    ay = ay_raw * ACC_SCALE
    az = az_raw * ACC_SCALE
    a_raw = np.array([ax - _b_acc[0], ay - _b_acc[1], az - _b_acc[2]])
    a_cal = _M_acc @ a_raw
    return a_cal[0], a_cal[1], a_cal[2]

def calibrate_gyro(gx_raw, gy_raw, gz_raw):
    """
    陀螺仪原始 ADC → 标定后 °/s
    输入：单点原始 ADC 值（gx_raw, gy_raw, gz_raw）
    返回：(gx_cal, gy_cal, gz_cal) 单位 °/s
    """
    gx = gx_raw * GYRO_SCALE
    gy = gy_raw * GYRO_SCALE
    gz = gz_raw * GYRO_SCALE
    return gx - _b_gyro[0], gy - _b_gyro[1], gz - _b_gyro[2]

def calibrate_mag(mx_raw, my_raw, mz_raw):
    """
    磁力计原始 LSB → 标定后 μT
    输入：单点原始 LSB 值（mx_raw, my_raw, mz_raw）
    返回：(mx_cal, my_cal, mz_cal) 单位 μT
    """
    mag_lsb = np.array([mx_raw, my_raw, mz_raw])
    centered = mag_lsb - _c_lsb           # 硬铁校正
    mag_norm = _A_lsb @ centered          # 软铁校正（归一化向量）
    mag_cal = mag_norm * B_NORM           # 缩放至地磁场强度
    return mag_cal[0], mag_cal[1], mag_cal[2]

# ================== 可选：重新加载参数 ==================
def reload_calib(calib_json_path):
    """如果需要动态切换标定文件，调用此函数更新全局参数"""
    global _M_acc, _b_acc, _b_gyro, _A_lsb, _c_lsb
    _M_acc, _b_acc, _b_gyro, _A_lsb, _c_lsb = _load_calib(calib_json_path)