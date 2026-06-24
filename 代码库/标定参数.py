import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import svd
import json, os

# ================== 加速度计六位置标定 ==================
def six_position_calibrate(file_dict):
    means = {}
    for pose, fname in file_dict.items():
        if not os.path.exists(fname):
            raise FileNotFoundError(f"加速度文件缺失: {fname}")
        data = np.loadtxt(fname, delimiter=',', skiprows=1)
        acc = data[:, 1:4] / 16384.0          # LSB → g (±2g量程)
        means[pose] = acc.mean(axis=0)
        print(f"{pose}: 均值 = {means[pose]} g")

    obs = np.array([means[k] for k in ['px','nx','py','ny','pz','nz']])
    def residuals(params):
        M = params[:9].reshape(3,3)
        b = params[9:12]
        cal = (M @ (obs - b).T).T
        return np.linalg.norm(cal, axis=1) - 1.0

    x0 = np.array([1,0,0, 0,1,0, 0,0,1, 0,0,0])
    res = least_squares(residuals, x0)
    M = res.x[:9].reshape(3,3)
    b = res.x[9:12]
    return M, b

# ================== 陀螺仪零偏 ==================
def gyro_bias_calibrate(static_file):
    if not os.path.exists(static_file):
        raise FileNotFoundError(f"陀螺仪静态文件缺失: {static_file}")
    data = np.loadtxt(static_file, delimiter=',', skiprows=1)
    gyro = data[:, 4:7] / 131.0            # LSB → °/s
    bias = gyro.mean(axis=0)
    return bias

# ================== 磁力计椭球拟合 ==================
def ellipsoid_fit(mag_file):
    if not os.path.exists(mag_file):
        return None, None  # 文件不存在，返回空
    data = np.loadtxt(mag_file, delimiter=',', skiprows=1)
    if data.shape[1] < 3:
        return None, None
    mag = data[:, :3]
    x, y, z = mag[:,0], mag[:,1], mag[:,2]
    D = np.column_stack([x**2, y**2, z**2,
                         2*x*y, 2*x*z, 2*y*z,
                         2*x, 2*y, 2*z])
    rhs = np.ones(len(x))
    U, S, Vt = svd(D, full_matrices=False)
    params = Vt.T @ np.diag(1/S) @ U.T @ rhs
    A_mat = np.array([[params[0], params[3], params[4]],
                      [params[3], params[1], params[5]],
                      [params[4], params[5], params[2]]])
    b_vec = params[6:9]
    center = -np.linalg.solve(A_mat, b_vec)
    eigvals, eigvecs = np.linalg.eigh(A_mat)
    eigvals = np.abs(eigvals)
    transform = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return center, transform

# ================== 主程序 ==================
if __name__ == "__main__":
    # ---------- 文件配置（按实际文件名修改）----------
    accel_files = {
        'px': 'accel_px.csv',
        'nx': 'accel_nx.csv',
        'py': 'accel_py.csv',
        'ny': 'accel_ny.csv',
        'pz': 'accel_pz.csv',
        'nz': 'accel_nz.csv',
    }
    gyro_file = 'gyro_static.csv'
    mag_file = 'mag_calib_raw1.csv'      # 若没有磁力计文件，会自动跳过
    mag_json_path = 'mag_calib.json'    # 可选：已有的磁力计标定JSON

    # ---------- 加速度计标定 ----------
    print("=== 加速度计六位置标定 ===")
    M, b = six_position_calibrate(accel_files)
    print(f"M:\n{M}\nbias: {b}")

    # ---------- 陀螺仪标定 ----------
    print("\n=== 陀螺仪零偏 ===")
    gyro_bias = gyro_bias_calibrate(gyro_file)
    print(f"gyro bias (°/s): {gyro_bias}")

    # ---------- 磁力计标定（自动选择来源）----------
    center, T = None, None
    # 尝试从 CSV 文件拟合
    if os.path.exists(mag_file):
        print(f"\n=== 磁力计椭球拟合（从 {mag_file}）===")
        center, T = ellipsoid_fit(mag_file)
        if center is not None:
            print(f"center (LSB): {center}")
            print(f"T:\n{T}")
        else:
            print("椭球拟合失败，检查数据质量。")
    # 如果 CSV 拟合失败，尝试读取已有 JSON
    if center is None and mag_json_path and os.path.exists(mag_json_path):
        print(f"\n=== 读取已有磁力计标定参数（从 {mag_json_path}）===")
        with open(mag_json_path, 'r') as f:
            old = json.load(f)
        mag_data = old.get("HMC5883L_magnetometer", {})
        if mag_data:
            center = np.array(mag_data.get("center_offset_c", []))
            T = np.array(mag_data.get("transform_matrix_A_half", []))
            print("已从JSON加载磁力计参数。")
        else:
            print("JSON中无磁力计数据。")

    # 若仍无磁力计参数，设为空列表
    if center is None or T is None:
        mag_params = {
            "transform_matrix_A_half": [],
            "center_offset_c": [],
            "unit": ""
        }
    else:
        mag_params = {
            "transform_matrix_A_half": T.tolist() if isinstance(T, np.ndarray) else T,
            "center_offset_c": center.tolist() if isinstance(center, np.ndarray) else center,
            "unit": "LSB (原始ADC值)"
        }

    # ---------- 生成完整 JSON ----------
    calib_all = {
        "hardware": {
            "IMU": "MPU6050 (GY-521)",
            "Magnetometer": "HMC5883L (GY-273)"
        },
        "MPU6050_accelerometer": {
            "calibration_method": "6-position Least Squares",
            "M_matrix": M.tolist(),
            "bias_vector": b.tolist(),
            "unit": "g"
        },
        "HMC5883L_magnetometer": {
            "calibration_method": "Ellipsoid Fitting (SVD)",
            **mag_params
        },
        "MPU6050_gyroscope": {
            "calibration_method": "Static Averaging",
            "bias_vector": gyro_bias.tolist(),
            "unit": "deg/s"
        }
    }

    with open('calib_params.json', 'w') as f:
        json.dump(calib_all, f, indent=2)
    print("\n✅ 标定参数已保存至 calib_params.json")