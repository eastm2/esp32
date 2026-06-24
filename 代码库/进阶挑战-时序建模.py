# d9_direction2_hmm_smoothed_fix.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from hmmlearn import hmm
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ================== 配置 ==================
FEAT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\features")
CAL_FEAT_PATH = FEAT_DIR / "feature_matrix_cal.csv"
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D9_advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFS_TOP6 = [
    "fusion_vib_energy", "fusion_grav_ratio_y", "fusion_acc_mag_std",
    "freq_gyro_z_band1_0.5-2Hz_energy_ratio", "fusion_pitch_rel_mean", "time_basic_acc_y_mean"
]
RANDOM_STATE = 42
N_CLASSES = 6
MAJORITY_WINDOW = 5
LAPLACE_SMOOTHING = 0.1  # 拉普拉斯平滑因子

# ================== 加载数据 ==================
df = pd.read_csv(CAL_FEAT_PATH)
X = df[SFS_TOP6].values
y = df["label"].values
subject_ids = df["subject_id"].values
print(f"数据：{len(df)} 窗口，{len(np.unique(subject_ids))} 名被试")

# ================== 1. SVM 预测 ==================
print("生成 RBF SVM LOSO 预测...")
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1, gamma="scale", probability=False, random_state=RANDOM_STATE))
])
logo = LeaveOneGroupOut()
y_pred_svm = cross_val_predict(svm_pipe, X, y, cv=logo, groups=subject_ids, n_jobs=-1)
acc_before = accuracy_score(y, y_pred_svm)
print(f"原始 SVM 准确率: {acc_before:.4f}")

# ================== 多数表决（已验证有效） ==================
def majority_vote(arr):
    return Counter(arr).most_common(1)[0][0]

def majority_vote_smoothing(seq, window_size=MAJORITY_WINDOW):
    n = len(seq)
    smoothed = np.zeros_like(seq)
    half = window_size // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = majority_vote(seq[start:end])
    return smoothed

def subject_wise_smooth(y_pred, subject_ids, func, **kwargs):
    y_sm = np.zeros_like(y_pred)
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        y_sm[mask] = func(y_pred[mask], **kwargs) if len(y_pred[mask]) > 1 else y_pred[mask]
    return y_sm

y_maj = subject_wise_smooth(y_pred_svm, subject_ids, majority_vote_smoothing)
acc_maj = accuracy_score(y, y_maj)
print(f"多数表决后准确率: {acc_maj:.4f}，提升 {acc_maj - acc_before:.4f}")

# ================== HMM 平滑（加入拉普拉斯平滑） ==================
print("\n构建 HMM 平滑（带拉普拉斯平滑）...")
y_hmm = np.zeros(len(y), dtype=int)

for test_subj in np.unique(subject_ids):
    train_mask = subject_ids != test_subj
    test_mask = subject_ids == test_subj
    if np.sum(test_mask) == 0:
        continue

    y_train_true = y[train_mask]
    y_train_pred = y_pred_svm[train_mask]
    subj_train = subject_ids[train_mask]

    # --- 转移计数（加平滑） ---
    trans_counts = np.full((N_CLASSES, N_CLASSES), LAPLACE_SMOOTHING)
    for s in np.unique(subj_train):
        idx = np.where(subj_train == s)[0]
        if len(idx) < 2:
            continue
        seq = y_train_true[idx]
        for t in range(len(seq) - 1):
            trans_counts[seq[t], seq[t+1]] += 1
    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # --- 发射概率（混淆矩阵）加平滑 ---
    cm = confusion_matrix(y_train_true, y_train_pred, labels=range(N_CLASSES))
    cm = cm.astype(float) + LAPLACE_SMOOTHING
    cm /= cm.sum(axis=1, keepdims=True)

    # --- HMM 解码 ---
    model = hmm.CategoricalHMM(n_components=N_CLASSES, random_state=RANDOM_STATE)
    model.startprob_ = np.full(N_CLASSES, 1.0 / N_CLASSES)
    model.transmat_ = trans_prob
    model.emissionprob_ = cm

    test_seq = y_pred_svm[test_mask].reshape(-1, 1)
    if len(test_seq) == 1:
        y_hmm[test_mask] = test_seq.flatten()
    else:
        _, smoothed = model.decode(test_seq, algorithm='viterbi')
        y_hmm[test_mask] = smoothed

acc_hmm = accuracy_score(y, y_hmm)
print(f"HMM 平滑后准确率: {acc_hmm:.4f}，提升 {acc_hmm - acc_before:.4f}")

# 保存结果
pd.DataFrame({
    "真实": y,
    "SVM原始": y_pred_svm,
    "多数表决": y_maj,
    "HMM平滑": y_hmm
}).to_csv(OUTPUT_DIR / "时序平滑结果_最终.csv", index=False, encoding="utf_8_sig")

print("\n====== 方向二 最终对比 ======")
print(f"原始 SVM  : {acc_before:.4f}")
print(f"多数表决  : {acc_maj:.4f} (+{(acc_maj-acc_before)*100:.2f}%)")
print(f"HMM 平滑  : {acc_hmm:.4f} (+{(acc_hmm-acc_before)*100:.2f}%)")