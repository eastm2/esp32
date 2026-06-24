# collect_sensors_separately.py
import serial
import time
from pathlib import Path
from datetime import datetime

# ================== 配置 ==================
SERIAL_PORT = "COM6"
BAUDRATE = 115200
OUTPUT_DIR = Path("collected_sensors")
OUTPUT_DIR.mkdir(exist_ok=True)

# ================== 串口连接 ==================
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
print(f"串口 {SERIAL_PORT} 已连接")

subject_id = input("请输入被试ID (整数): ").strip()

# 选择要采集的传感器
print("\n请选择要采集的传感器：")
print("1 - 仅 MPU6050 (加速度计 + 陀螺仪)")
print("2 - 仅 HMC5883L (磁力计)")
print("3 - 两个都采 (分别保存)")
choice = input("输入 1/2/3: ").strip()

collect_mpu = choice in ('1', '3')
collect_hmc = choice in ('2', '3')

if not collect_mpu and not collect_hmc:
    print("未选择任何传感器，退出。")
    ser.close()
    exit()

print("\n活动标签: 0-静坐 1-站立 2-行走 3-跑步 4-上楼 5-下楼")

# 缓存与文件句柄
rows_mpu = []    # [timestamp, label, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw]
rows_hmc = []    # [timestamp, label, mx_raw, my_raw, mz_raw]
csv_mpu = None
csv_hmc = None
record_t0 = None

def start_new_label(label):
    global record_t0, csv_mpu, csv_hmc, rows_mpu, rows_hmc
    rows_mpu = []
    rows_hmc = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sensor_tag = ""
    if collect_mpu:
        csv_mpu = OUTPUT_DIR / f"sub{subject_id}_label{label}_mpu_raw_{timestamp}.csv"
        with open(csv_mpu, 'w') as f:
            f.write("timestamp_ms,label,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw\n")
        sensor_tag += "MPU6050"
    if collect_hmc:
        csv_hmc = OUTPUT_DIR / f"sub{subject_id}_label{label}_hmc_raw_{timestamp}.csv"
        with open(csv_hmc, 'w') as f:
            f.write("timestamp_ms,label,mx_raw,my_raw,mz_raw\n")
        sensor_tag += (" & " if sensor_tag else "") + "HMC5883L"
    print(f"开始采集活动 {label} ({sensor_tag})，按 Ctrl+C 停止。")
    record_t0 = time.time()

def flush_buffers():
    global rows_mpu, rows_hmc
    if collect_mpu and csv_mpu and rows_mpu:
        with open(csv_mpu, 'a') as f:
            for row in rows_mpu:
                f.write(','.join(map(str, row)) + '\n')
        rows_mpu = []
    if collect_hmc and csv_hmc and rows_hmc:
        with open(csv_hmc, 'a') as f:
            for row in rows_hmc:
                f.write(','.join(map(str, row)) + '\n')
        rows_hmc = []

try:
    while True:
        label_str = input("\n请输入活动标签 (0-5) 或 'q' 结束: ").strip()
        if label_str.lower() == 'q':
            break
        try:
            label = int(label_str)
            if label not in range(6):
                print("标签需在 0-5 之间")
                continue
        except:
            print("无效输入")
            continue

        start_new_label(label)

        print("按 Ctrl+C 停止当前活动...")
        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                try:
                    parts = list(map(float, line.split(',')))
                    if len(parts) != 9:
                        continue
                    ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, mx_raw, my_raw, mz_raw = parts
                except:
                    continue

                ts_ms = int((time.time() - record_t0) * 1000)

                if collect_mpu:
                    rows_mpu.append([ts_ms, label,
                                     int(ax_raw), int(ay_raw), int(az_raw),
                                     int(gx_raw), int(gy_raw), int(gz_raw)])
                if collect_hmc:
                    rows_hmc.append([ts_ms, label,
                                     int(mx_raw), int(my_raw), int(mz_raw)])

                # 每 50 行写入一次
                if max(len(rows_mpu), len(rows_hmc)) >= 50:
                    flush_buffers()

        except KeyboardInterrupt:
            print(f"\n活动 {label} 采集中断。")
            flush_buffers()
            print(f"数据已保存。")

except KeyboardInterrupt:
    print("\n完全退出。")

finally:
    ser.close()
    print("串口已关闭。")