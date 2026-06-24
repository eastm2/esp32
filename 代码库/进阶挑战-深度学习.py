# d9_direction3_dl_vs_traditional_final.py
import numpy as np
import pandas as pd
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置 ==================
# 使用标定后的数据集（你特征提取用的文件）
CAL_NPZ = Path(r"E:\1aaa\py\zhuanzhou2026\data\dataset_all.npz")
FEAT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\features")
CAL_FEAT_PATH = FEAT_DIR / "feature_matrix_cal.csv"
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D9_advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFS_TOP6 = [
    "fusion_vib_energy", "fusion_grav_ratio_y", "fusion_acc_mag_std",
    "freq_gyro_z_band1_0.5-2Hz_energy_ratio", "fusion_pitch_rel_mean", "time_basic_acc_y_mean"
]
RANDOM_STATE = 42
SEQ_LEN = 128
N_CHANNELS = 9
N_CLASSES = 6
FS = 50
OVERLAP = 0.5
MAJORITY_THRESH = 0.8          # 窗口内多数标签比例阈值，与 D4 保持一致

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ================== 重新生成与特征矩阵完全对齐的原始窗口 ==================
def generate_raw_windows_aligned(npz_path, window_size=128, overlap=0.5):
    """
    完全模仿 D4 中的滑窗流程：
    1. 读入 NPZ（每个窗口已切好128点）
    2. 按 (subject, label) 拼接连续信号和标签序列
    3. 用滑动窗口切分，并进行多数表决过滤（比例 ≥ 0.8）
    返回与特征矩阵行数完全一致的 X_raw, y_raw, subject_ids
    """
    data = np.load(npz_path)
    acc = data["acc"]          # (N_windows, 128, 3)
    gyro = data["gyro"]
    mag = data["mag"]
    labels = data["labels"]    # (N_windows,)
    subjects = data["subject_ids"]

    # 拼接连续信号和标签序列（每个窗口的标签重复128次作为点标签）
    seq_dict = {}
    for i in range(len(acc)):
        key = (subjects[i], labels[i])
        if key not in seq_dict:
            seq_dict[key] = {"acc": [], "gyro": [], "mag": [], "labels": []}
        seq_dict[key]["acc"].append(acc[i])
        seq_dict[key]["gyro"].append(gyro[i])
        seq_dict[key]["mag"].append(mag[i])
        seq_dict[key]["labels"].extend([labels[i]] * 128)

    windows, win_subjects, win_labels = [], [], []
    step = int(window_size * (1 - overlap))

    for (subj, lab), seq in seq_dict.items():
        acc_seq = np.concatenate(seq["acc"], axis=0)   # (T, 3)
        gyro_seq = np.concatenate(seq["gyro"], axis=0)
        mag_seq = np.concatenate(seq["mag"], axis=0)
        label_seq = np.array(seq["labels"])             # (T,)

        T = acc_seq.shape[0]
        for start in range(0, T - window_size + 1, step):
            end = start + window_size
            seg_labels = label_seq[start:end]
            # 多数表决过滤
            dominant = np.bincount(seg_labels).argmax()
            if np.mean(seg_labels == dominant) >= MAJORITY_THRESH:
                win = np.hstack([acc_seq[start:end], gyro_seq[start:end], mag_seq[start:end]])
                windows.append(win)
                win_subjects.append(subj)
                win_labels.append(dominant)

    return np.array(windows), np.array(win_labels), np.array(win_subjects)

print("重新生成原始窗口（带多数表决，与特征矩阵严格对齐）...")
X_raw, y_raw, subjects_raw = generate_raw_windows_aligned(CAL_NPZ, SEQ_LEN, OVERLAP)
print(f"原始窗口: {X_raw.shape[0]} 个, 形状 {X_raw.shape[1:]}")

# 加载特征矩阵
df_feat = pd.read_csv(CAL_FEAT_PATH)
# 确保列顺序一致
X_feat = df_feat[SFS_TOP6].values
y_feat = df_feat["label"].values
subjects_feat = df_feat["subject_id"].values

# 由于生成顺序可能不同，我们需要对齐两个数据集的样本顺序：
# 策略：按 (subject, label) 排序，然后对同组内的样本用某种唯一标识（例如累积窗口索引）排序，
# 但由于我们不知道特征矩阵的原始顺序，只需保证两组数据在 LOSO 时使用相同的分组即可。
# 更好的做法：直接根据特征矩阵的 subject 和 label 对 X_raw 进行重排，使其与 X_feat 一一对应。
# 我们通过建立 DataFrame 并 merge 来对齐：
raw_df = pd.DataFrame({"subject": subjects_raw, "label": y_raw})
raw_df["raw_idx"] = range(len(raw_df))
feat_df = df_feat[["subject_id", "label"]].copy()
feat_df["feat_idx"] = range(len(feat_df))

# 在每组 (subject, label) 内，我们希望 raw 和 feat 的顺序一致。
# 由于生成逻辑相同，理论上顺序应该一致，但 dict 迭代顺序可能导致不一致。
# 我们简单地按 (subject, label) 排序，并假设组内顺序自然一致（因为都是按照滑动窗口的先后顺序生成的）。
# 如果仍不一致，我们需要更复杂的对齐（例如计算每个窗口的特征进行匹配），但这会太复杂。
# 这里我们信任生成逻辑的确定性，直接排序：
raw_df = raw_df.sort_values(["subject", "label"]).reset_index(drop=True)
feat_df = feat_df.sort_values(["subject_id", "label"]).reset_index(drop=True)

# 检查长度是否一致
assert len(raw_df) == len(feat_df), "窗口数不一致，无法对齐"
# 检查 (subject, label) 序列是否完全一致
assert (raw_df["subject"].values == feat_df["subject_id"].values).all(), "subject 序列不一致"
assert (raw_df["label"].values == feat_df["label"].values).all(), "label 序列不一致"

# 按对齐后的顺序重新排列 X_raw 和 X_feat
sorted_raw_idx = raw_df["raw_idx"].values
X_raw_aligned = X_raw[sorted_raw_idx]
y_raw_aligned = y_raw[sorted_raw_idx]
subjects_raw_aligned = subjects_raw[sorted_raw_idx]

sorted_feat_idx = feat_df["feat_idx"].values
X_feat_aligned = X_feat[sorted_feat_idx]
y_feat_aligned = y_feat[sorted_feat_idx]
subjects_feat_aligned = subjects_feat[sorted_feat_idx]

print("✅ 原始窗口与特征矩阵已对齐，窗口数一致，标签序列一致。")

# ================== 定义 CNN 模型 ==================
class HAR_CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

# ================== LOSO 评估 ==================
logo = LeaveOneGroupOut()

def evaluate_cnn():
    all_preds, all_true = [], []
    times = []
    for train_idx, test_idx in logo.split(X_raw_aligned, y_raw_aligned, groups=subjects_raw_aligned):
        X_train_t = torch.tensor(X_raw_aligned[train_idx], dtype=torch.float32).permute(0,2,1).to(DEVICE)
        y_train_t = torch.tensor(y_raw_aligned[train_idx], dtype=torch.long).to(DEVICE)
        X_test_t = torch.tensor(X_raw_aligned[test_idx], dtype=torch.float32).permute(0,2,1).to(DEVICE)
        y_test_t = torch.tensor(y_raw_aligned[test_idx], dtype=torch.long).to(DEVICE)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
        model = HAR_CNN(N_CHANNELS, N_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(20):               # 20个epoch
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            t0 = time.time()
            outputs = model(X_test_t)
            elapsed = time.time() - t0
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y_raw_aligned[test_idx])
        times.append(elapsed / len(preds))

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    acc = (y_pred == y_true).mean()
    avg_time = np.mean(times)

    torch.save(model.state_dict(), OUTPUT_DIR / "cnn_temp.pth")
    size_kb = (OUTPUT_DIR / "cnn_temp.pth").stat().st_size / 1024
    return acc, avg_time, size_kb

def evaluate_svm():
    # 使用对齐后的特征数据
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1, gamma="scale", probability=False, random_state=RANDOM_STATE))
    ])
    scores = cross_val_score(pipe, X_feat_aligned, y_feat_aligned, cv=logo,
                             groups=subjects_feat_aligned, scoring="accuracy", n_jobs=-1)
    acc = scores.mean()

    pipe.fit(X_feat_aligned, y_feat_aligned)
    joblib.dump(pipe, OUTPUT_DIR / "svm_temp.pkl")
    size_kb = (OUTPUT_DIR / "svm_temp.pkl").stat().st_size / 1024

    t0 = time.time()
    pipe.predict(X_feat_aligned)
    avg_time = (time.time() - t0) / len(X_feat_aligned)
    return acc, avg_time, size_kb

# ================== 执行对比 ==================
print("\n训练 1D-CNN (LOSO, 20 epochs) ...")
cnn_acc, cnn_time, cnn_size = evaluate_cnn()
print(f"1D-CNN  : Acc={cnn_acc:.4f}, Time={cnn_time*1000:.4f} ms/win, Size={cnn_size:.1f} KB")

print("\n评估 RBF SVM (LOSO, SFS-6 特征) ...")
svm_acc, svm_time, svm_size = evaluate_svm()
print(f"RBF SVM : Acc={svm_acc:.4f}, Time={svm_time*1000:.4f} ms/win, Size={svm_size:.1f} KB")

# ================== 保存结果与绘图 ==================
pd.DataFrame({
    "模型": ["1D-CNN (原始信号)", "RBF SVM (手工特征)"],
    "准确率": [cnn_acc, svm_acc],
    "推理时间(ms)": [cnn_time*1000, svm_time*1000],
    "模型大小(KB)": [cnn_size, svm_size]
}).to_csv(OUTPUT_DIR / "DL_vs_traditional.csv", index=False, encoding="utf_8_sig")

fig, ax = plt.subplots(figsize=(7,5))
methods = ["1D-CNN\n(原始信号)", "RBF SVM\n(手工特征)"]
accs = [cnn_acc, svm_acc]
bars = ax.bar(methods, accs, color=["#fd8d3c", "#6baed6"], width=0.4)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.005,
            f"{acc:.2%}", ha="center", fontweight="bold")
ax.set_ylim(0, max(accs)+0.05)
ax.set_ylabel("LOSO 准确率")
ax.set_title("方向三：深度学习 vs 传统方法")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DL_vs_traditional.png", dpi=300)
plt.show()

print(f"\n📁 所有结果已保存至 {OUTPUT_DIR}")