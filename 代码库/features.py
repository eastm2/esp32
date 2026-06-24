import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy import stats
import pywt
import warnings
warnings.filterwarnings("ignore")

# ================== 配置 ==================
CONFIG = {
    "FS": 50,
    "WINDOW_SIZE": 128,
    "OVERLAP": 0.5,
    "GRAVITY_CUTOFF": 0.3,
    "BUTTER_ORDER": 4,
    "ENABLE_TIME_BASIC": True,
    "ENABLE_TIME_ADVANCED": True,
    "ENABLE_FREQ_FEATURES": True,
    "ENABLE_TIMEFREQ_FEATURES": True,
    "ENABLE_FUSION_FEATURES": True,
    "ENABLE_MAG_FEATURES": True,
    "FREQ_BANDS": [(0.5, 2), (2, 5), (5, 10), (10, 25)],
    "OUTPUT_DIR": Path(r"E:\1aaa\py\zhuanzhou2026\data\features"),
}
CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

# ================== 工具函数 ==================
def separate_gravity(acc, fs, cutoff=0.3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    gravity = signal.filtfilt(b, a, acc, axis=0)
    motion = acc - gravity
    return gravity, motion

def calculate_raw_attitude(gravity_win, mag_win=None):
    gx, gy, gz = gravity_win[:, 0], gravity_win[:, 1], gravity_win[:, 2]
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-10
    pitch_rad = np.arcsin(np.clip(gx / g_mag, -1.0, 1.0))
    roll_rad = np.arcsin(np.clip(gz / g_mag, -1.0, 1.0))
    pitch = np.degrees(pitch_rad)
    roll = np.degrees(roll_rad)
    heading = np.zeros_like(pitch)
    if mag_win is not None and np.any(mag_win):
        mx, my, mz = mag_win[:, 0], mag_win[:, 1], mag_win[:, 2]
        mx_comp = mx * np.cos(pitch_rad) + mz * np.sin(pitch_rad)
        my_comp = (mx * np.sin(pitch_rad) * np.sin(roll_rad) +
                   my * np.cos(roll_rad) -
                   mz * np.cos(pitch_rad) * np.sin(roll_rad))
        heading_rad = np.arctan2(-my_comp, mx_comp)
        heading = np.degrees(heading_rad) % 360
    return pitch, roll, heading

def sliding_window(data, labels, has_mag=None, window_size=128, overlap=0.5):
    step = int(window_size * (1 - overlap))
    num_samples = data.shape[0]
    windows, win_labels, win_has_mag = [], [], []
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        window = data[start:end]
        label_seg = labels[start:end]
        mag_seg = has_mag[start:end] if has_mag is not None else None
        dominant_label = np.bincount(label_seg).argmax()
        if np.mean(label_seg == dominant_label) >= 0.8:
            windows.append(window)
            win_labels.append(dominant_label)
            if mag_seg is not None:
                win_has_mag.append(np.all(mag_seg))
            else:
                win_has_mag.append(False)
    return np.array(windows), np.array(win_labels), np.array(win_has_mag)

def compute_subject_baselines(raw_meta_df):
    baselines = {}
    sit_df = raw_meta_df[raw_meta_df["label"] == 0]
    for subj in sit_df["subject_id"].unique():
        subj_sit = sit_df[sit_df["subject_id"] == subj]
        pitch_base = subj_sit["fusion_pitch_raw"].mean()
        roll_base = subj_sit["fusion_roll_raw"].mean()
        baselines[subj] = {"pitch_base": pitch_base, "roll_base": roll_base}
        print(f"✅ 受试者{subj} 姿态基准：俯仰={pitch_base:.2f}°，横滚={roll_base:.2f}°")
    return baselines

# ================== 特征提取器 ==================
class FeatureExtractor:
    def __init__(self, fs=50, subject_baselines=None):
        self.fs = fs
        self.subject_baselines = subject_baselines
        self.feature_definitions = {}   # 仅保留内部记录，不输出文件
        nyq = 0.5 * fs
        self.vib_b, self.vib_a = signal.butter(2, [0.5/nyq, 3.0/nyq], btype='band')

    def _add_feature_def(self, fname, full_name, definition, group, sensor):
        if fname not in self.feature_definitions:
            self.feature_definitions[fname] = {
                "full_name": full_name, "definition": definition,
                "group": group, "sensor": sensor,
                "physical_meaning": ""
            }

    # 1. 基础时域特征
    def extract_time_basic(self, signal_3axis, sensor_prefix="acc"):
        features = []
        axis_names = ["x", "y", "z"]
        for i in range(3):
            sig = signal_3axis[:, i]
            ax = axis_names[i]
            prefix = f"{sensor_prefix}_{ax}"
            features.extend([
                np.mean(sig), np.std(sig), np.var(sig),
                np.sqrt(np.mean(sig ** 2)),
                np.max(sig) - np.min(sig),
                np.max(sig), np.min(sig), np.median(sig),
                stats.skew(sig), stats.kurtosis(sig),
                np.sum(sig[:-1] * sig[1:] < 0) / len(sig),
                np.mean(np.abs(sig))
            ])
            # 记录特征定义（保留内部使用，不输出文件）
            for fname in [
                f"time_basic_{prefix}_mean", f"time_basic_{prefix}_std", f"time_basic_{prefix}_var",
                f"time_basic_{prefix}_rms", f"time_basic_{prefix}_peak2peak", f"time_basic_{prefix}_max",
                f"time_basic_{prefix}_min", f"time_basic_{prefix}_median", f"time_basic_{prefix}_skewness",
                f"time_basic_{prefix}_kurtosis", f"time_basic_{prefix}_zerocross_rate", f"time_basic_{prefix}_sma"
            ]:
                self._add_feature_def(fname, "", "", "时域基础", f"{prefix}")
        return np.array(features)

    # 2. 高阶时域特征（含波形因子）
    def extract_time_advanced(self, signal_3axis, sensor_prefix="acc"):
        features = []
        axis_names = ["x", "y", "z"]
        for i in range(3):
            sig = signal_3axis[:, i]
            ax = axis_names[i]
            prefix = f"{sensor_prefix}_{ax}"
            autocorr = np.corrcoef(sig[:-1], sig[1:])[0, 1] if len(sig) > 1 else 0
            q75, q25 = np.percentile(sig, [75, 25])
            iqr_val = q75 - q25
            mav_val = np.mean(np.abs(sig - np.mean(sig)))
            rms = np.sqrt(np.mean(sig**2))
            ma = np.mean(np.abs(sig))
            pk = np.max(np.abs(sig))
            features.extend([
                autocorr, iqr_val, mav_val,
                rms / (ma + 1e-8),          # shape factor
                pk / (rms + 1e-8),          # crest factor
                pk / (ma + 1e-8),           # impulse factor
                pk / (np.mean(np.sqrt(np.abs(sig)))**2 + 1e-8)  # clearance factor
            ])
            for fname in [
                f"time_adv_{prefix}_autocorr_lag1", f"time_adv_{prefix}_iqr",
                f"time_adv_{prefix}_mean_abs_dev", f"time_adv_{prefix}_shape_factor",
                f"time_adv_{prefix}_crest_factor", f"time_adv_{prefix}_impulse_factor",
                f"time_adv_{prefix}_clearance_factor"
            ]:
                self._add_feature_def(fname, "", "", "时域高阶", f"{prefix}")
        return np.array(features)

    # 3. 频域特征
    def extract_freq_features(self, signal_3axis, sensor_prefix="acc"):
        features = []
        axis_names = ["x", "y", "z"]
        n_fft = len(signal_3axis)
        freqs = np.fft.rfftfreq(n_fft, d=1 / self.fs)
        for i in range(3):
            sig = signal_3axis[:, i]
            ax = axis_names[i]
            prefix = f"{sensor_prefix}_{ax}"
            fft_vals = np.fft.rfft(sig)
            psd = np.abs(fft_vals) ** 2 / (self.fs * n_fft)
            main_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
            centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features.extend([main_freq, centroid, spectral_entropy])
            for f_low, f_high in CONFIG["FREQ_BANDS"]:
                band_energy = np.sum(psd[(freqs >= f_low) & (freqs <= f_high)]) / (np.sum(psd) + 1e-10)
                features.append(band_energy)
            cumsum_psd = np.cumsum(psd)
            half_power_idx = np.where(cumsum_psd >= 0.5 * cumsum_psd[-1])[0]
            bandwidth = freqs[half_power_idx[0]] if len(half_power_idx) > 0 else 0
            rms_freq = np.sqrt(np.sum(freqs ** 2 * psd) / (np.sum(psd) + 1e-10))
            features.extend([bandwidth, rms_freq])
            geom_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arith_mean = np.mean(psd + 1e-10)
            flatness = geom_mean / (arith_mean + 1e-10)
            log_freqs = np.log10(freqs + 1e-10)
            log_psd = np.log10(psd + 1e-10)
            slope, _ = np.polyfit(log_freqs, log_psd, 1) if len(log_freqs) > 1 else (0.0, 0.0)
            features.extend([flatness, slope])
            for fname in [
                f"freq_{prefix}_main_freq", f"freq_{prefix}_spectral_centroid", f"freq_{prefix}_spectral_entropy",
                f"freq_{prefix}_3db_bandwidth", f"freq_{prefix}_rms_freq",
                f"freq_{prefix}_spectral_flatness", f"freq_{prefix}_spectral_slope"
            ]:
                self._add_feature_def(fname, "", "", "频域", f"{prefix}")
            for b_idx, _ in enumerate(CONFIG["FREQ_BANDS"]):
                self._add_feature_def(f"freq_{prefix}_band{b_idx+1}", "", "", "频域", f"{prefix}")
        return np.array(features)

    # 4. 时频特征（CWT）
    def extract_timefreq_features(self, signal_3axis, sensor_prefix="acc"):
        features = []
        axis_names = ["x", "y", "z"]
        scales = np.arange(1, 31)
        wavelet = "morl"
        for i in range(3):
            sig = signal_3axis[:, i]
            prefix = f"{sensor_prefix}_{axis_names[i]}"
            coef, freqs = pywt.cwt(sig, scales, wavelet, sampling_period=1 / self.fs)
            energy_mean = np.mean(np.abs(coef) ** 2)
            coef_norm = np.abs(coef) ** 2 / (np.sum(np.abs(coef) ** 2) + 1e-10)
            cwt_entropy = -np.sum(coef_norm * np.log2(coef_norm + 1e-10))
            main_scale_idx = np.argmax(np.mean(np.abs(coef) ** 2, axis=1))
            main_scale_energy = np.mean(np.abs(coef[main_scale_idx, :]) ** 2)
            main_scale_ratio = main_scale_energy / (np.sum(np.abs(coef) ** 2) + 1e-10)
            features.extend([energy_mean, cwt_entropy, main_scale_ratio])
            for fname in [
                f"timefreq_{prefix}_cwt_energy_mean", f"timefreq_{prefix}_cwt_entropy",
                f"timefreq_{prefix}_main_scale_energy_ratio"
            ]:
                self._add_feature_def(fname, "", "", "时频", f"{prefix}")
        return np.array(features)

    # 5. 融合特征
    def extract_fusion_features(self, acc_motion, gyro, mag=None, gravity_win=None, subject_id=None):
        features = []
        ax, ay, az = acc_motion[:, 0], acc_motion[:, 1], acc_motion[:, 2]
        gx, gy_, gz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

        corr_ax_ay = np.corrcoef(ax, ay)[0, 1] if len(ax) > 1 else 0
        corr_ax_az = np.corrcoef(ax, az)[0, 1] if len(ax) > 1 else 0
        corr_ay_az = np.corrcoef(ay, az)[0, 1] if len(ay) > 1 else 0
        features.extend([corr_ax_ay, corr_ax_az, corr_ay_az])

        for acc_sig, acc_axis in zip([ax, ay, az], ["ax", "ay", "az"]):
            for gyro_sig, gyro_axis in zip([gx, gy_, gz], ["gx", "gy", "gz"]):
                corr = np.corrcoef(acc_sig, gyro_sig)[0, 1] if len(acc_sig) > 1 else 0
                features.append(corr)

        acc_mag = np.sqrt(np.sum(acc_motion ** 2, axis=1))
        features.extend([np.mean(acc_mag), np.std(acc_mag)])
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
        features.extend([np.mean(gyro_mag), np.std(gyro_mag)])

        jerk = np.diff(acc_motion, axis=0)
        jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
        features.extend([np.mean(jerk_mag), np.std(jerk_mag)])

        try:
            from pyentrp import entropy as ent
            perm_entropy = ent.permutation_entropy(acc_mag, order=3, delay=1, normalize=True)
        except ImportError:
            perm_entropy = 0.0
        features.append(perm_entropy)

        vib = signal.filtfilt(self.vib_b, self.vib_a, acc_motion, axis=0)
        vib_energy = np.mean(np.sum(vib**2, axis=1))
        features.append(vib_energy)

        if gravity_win is not None and subject_id is not None and self.subject_baselines:
            mag_for_att = mag if (CONFIG["ENABLE_MAG_FEATURES"] and mag is not None and np.any(mag)) else None
            pitch_raw, roll_raw, heading_raw = calculate_raw_attitude(gravity_win, mag_for_att)
            base = self.subject_baselines.get(subject_id)
            if base:
                pitch_rel = -(pitch_raw - base["pitch_base"])
                roll_rel = roll_raw - base["roll_base"]
                features.extend([np.mean(pitch_rel), np.std(pitch_rel),
                                 np.max(pitch_rel) - np.min(pitch_rel),
                                 np.mean(np.abs(pitch_rel - np.median(pitch_rel)))])
                features.extend([np.mean(roll_rel), np.std(roll_rel),
                                 np.max(roll_rel) - np.min(roll_rel),
                                 np.mean(np.abs(roll_rel - np.median(roll_rel)))])

                gx_g, gy_g, gz_g = gravity_win[:,0], gravity_win[:,1], gravity_win[:,2]
                g_mag = np.sqrt(gx_g**2 + gy_g**2 + gz_g**2) + 1e-8
                grav_ratio_x = np.mean(gx_g / g_mag)
                grav_ratio_y = np.mean(gy_g / g_mag)
                grav_ratio_z = np.mean(gz_g / g_mag)
                features.extend([grav_ratio_x, grav_ratio_y, grav_ratio_z])

                horiz_proj = np.arctan2(gz_g, gx_g)
                features.extend([np.mean(horiz_proj), np.std(horiz_proj)])

                fft_pitch = np.abs(np.fft.rfft(pitch_rel))
                freqs_pitch = np.fft.rfftfreq(len(pitch_rel), d=1/self.fs)
                low_mask = (freqs_pitch >= 0.1) & (freqs_pitch <= 0.5)
                low_energy = np.sum(fft_pitch[low_mask]**2) / (np.sum(fft_pitch**2) + 1e-8)
                features.append(low_energy)

                if mag_for_att is not None:
                    heading_std = np.std(heading_raw)
                    heading_range = np.max(heading_raw) - np.min(heading_raw)
                    features.extend([heading_std, heading_range])
                else:
                    features.extend([0, 0])

                ratio_xz = (gx_g / g_mag) / (gz_g / g_mag + 1e-8)
                features.append(np.mean(ratio_xz))
                inter_pitch_roll = np.mean(pitch_rel * roll_rel)
                features.append(inter_pitch_roll)
            else:
                features.extend([0]*20)
        else:
            features.extend([0]*20)

        return np.array(features)

    def extract_single_window(self, acc_motion, gyro, mag=None, gravity_win=None, subject_id=None):
        feats = []
        if CONFIG["ENABLE_TIME_BASIC"]:
            feats.append(self.extract_time_basic(acc_motion, "acc"))
        if CONFIG["ENABLE_TIME_ADVANCED"]:
            feats.append(self.extract_time_advanced(acc_motion, "acc"))
        if CONFIG["ENABLE_FREQ_FEATURES"]:
            feats.append(self.extract_freq_features(acc_motion, "acc"))
            feats.append(self.extract_freq_features(gyro, "gyro"))
            if CONFIG["ENABLE_MAG_FEATURES"] and mag is not None and np.any(mag):
                feats.append(self.extract_freq_features(mag, "mag"))
        if CONFIG["ENABLE_TIMEFREQ_FEATURES"]:
            feats.append(self.extract_timefreq_features(acc_motion, "acc"))
            feats.append(self.extract_timefreq_features(gyro, "gyro"))
        if CONFIG["ENABLE_FUSION_FEATURES"]:
            mag_input = mag if (CONFIG["ENABLE_MAG_FEATURES"] and mag is not None) else None
            feats.append(self.extract_fusion_features(acc_motion, gyro, mag_input, gravity_win, subject_id))
        return np.concatenate(feats) if feats else np.array([])


# ================== 主流程 ==================
def process_dataset(npz_path, output_prefix, is_raw=False):
    print(f"\n{'='*60}\n处理数据集：{npz_path}\n{'='*60}")
    data = np.load(npz_path)
    acc = data["acc"]
    gyro = data["gyro"]
    mag = data["mag"]
    labels = data["labels"]
    subject_ids = data["subject_ids"]
    window_indices = data["window_indices"]
    has_mag = data["has_mag"]

    if is_raw:
        LSB_PER_UT = 10.9
        mag = mag / LSB_PER_UT
        print("   🔄 原始数据磁力计单位已从 LSB 转换为 μT")

    print(f"✅ 加载数据：{len(acc)}个原始窗口")
    print(f"   活动类别：{np.unique(labels)}")
    print(f"   受试者：{np.unique(subject_ids)}")
    print(f"   有效磁力计窗口：{np.sum(has_mag)}/{len(has_mag)}")

    print("\n🔄 拼接连续信号...")
    continuous_data = {}
    sort_idx = np.lexsort((window_indices, labels, subject_ids))
    acc_sorted = acc[sort_idx]; gyro_sorted = gyro[sort_idx]; mag_sorted = mag[sort_idx]
    labels_sorted = labels[sort_idx]; subject_ids_sorted = subject_ids[sort_idx]
    has_mag_sorted = has_mag[sort_idx]
    prev_key = None
    for i in range(len(acc_sorted)):
        key = (subject_ids_sorted[i], labels_sorted[i])
        if key != prev_key:
            if key not in continuous_data:
                continuous_data[key] = {"acc": [], "gyro": [], "mag": [], "labels": [], "has_mag": []}
            prev_key = key
        continuous_data[key]["acc"].append(acc_sorted[i])
        continuous_data[key]["gyro"].append(gyro_sorted[i])
        continuous_data[key]["mag"].append(mag_sorted[i])
        continuous_data[key]["labels"].extend([labels_sorted[i]] * 128)
        continuous_data[key]["has_mag"].extend([has_mag_sorted[i]] * 128)
    for key in continuous_data:
        continuous_data[key]["acc"] = np.vstack(continuous_data[key]["acc"])
        continuous_data[key]["gyro"] = np.vstack(continuous_data[key]["gyro"])
        continuous_data[key]["mag"] = np.vstack(continuous_data[key]["mag"])
        continuous_data[key]["labels"] = np.array(continuous_data[key]["labels"])
        continuous_data[key]["has_mag"] = np.array(continuous_data[key]["has_mag"])
    print(f"✅ 拼接完成：{len(continuous_data)}个连续信号片段")

    step = int(CONFIG["WINDOW_SIZE"] * (1 - CONFIG["OVERLAP"]))
    raw_meta = []
    for (subj_id, label), cd in continuous_data.items():
        gravity, acc_motion = separate_gravity(cd["acc"], CONFIG["FS"])
        win_gravity, _, _ = sliding_window(gravity, cd["labels"], window_size=CONFIG["WINDOW_SIZE"], overlap=CONFIG["OVERLAP"])
        for win_idx, gwin in enumerate(win_gravity):
            pitch_raw, roll_raw, _ = calculate_raw_attitude(gwin)
            raw_meta.append({"subject_id": subj_id, "label": label,
                             "fusion_pitch_raw": np.mean(pitch_raw),
                             "fusion_roll_raw": np.mean(roll_raw)})
    raw_df = pd.DataFrame(raw_meta)
    subject_baselines = compute_subject_baselines(raw_df)

    extractor = FeatureExtractor(CONFIG["FS"], subject_baselines)
    all_features, all_meta = [], []
    for (subj_id, label), cd in continuous_data.items():
        print(f"  受试者{subj_id}，活动{label}，长度{len(cd['acc'])}点")
        gravity, acc_motion = separate_gravity(cd["acc"], CONFIG["FS"])
        win_acc, win_labels, win_has_mag = sliding_window(acc_motion, cd["labels"], cd["has_mag"])
        win_gyro, _, _ = sliding_window(cd["gyro"], cd["labels"])
        win_mag, _, _ = sliding_window(cd["mag"], cd["labels"])
        win_gravity, _, _ = sliding_window(gravity, cd["labels"])
        print(f"      生成{len(win_acc)}个重叠窗口")
        for win_idx in range(len(win_acc)):
            feat = extractor.extract_single_window(
                win_acc[win_idx], win_gyro[win_idx],
                win_mag[win_idx] if CONFIG["ENABLE_MAG_FEATURES"] else None,
                win_gravity[win_idx], subj_id
            )
            all_features.append(feat)
            all_meta.append({"subject_id": subj_id, "label": win_labels[win_idx],
                             "window_idx": win_idx, "has_mag": win_has_mag[win_idx]})

    feature_matrix = np.array(all_features)
    meta_df = pd.DataFrame(all_meta)
    feature_cols = list(extractor.feature_definitions.keys())
    if feature_matrix.shape[1] != len(feature_cols):
        raise ValueError(f"列数不匹配：特征矩阵{feature_matrix.shape[1]} vs 定义{len(feature_cols)}")
    result_df = pd.concat([meta_df[["subject_id", "label", "window_idx", "has_mag"]],
                           pd.DataFrame(feature_matrix, columns=feature_cols)], axis=1)
    out_csv = CONFIG["OUTPUT_DIR"] / f"feature_matrix_{output_prefix}.csv"
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 特征矩阵已保存：{out_csv}，样本数{len(result_df)}，特征数{len(feature_cols)}")
    return result_df, extractor

if __name__ == "__main__":
    RAW_NPZ = Path(r"E:\1aaa\py\zhuanzhou2026\data\dataset_raw_all.npz")
    if RAW_NPZ.exists():
        process_dataset(RAW_NPZ, "raw", is_raw=True)

    CAL_NPZ = Path(r"E:\1aaa\py\zhuanzhou2026\data\dataset_all.npz")
    if CAL_NPZ.exists():
        process_dataset(CAL_NPZ, "cal", is_raw=False)

    print("\n🎉 D4 特征提取完成！")