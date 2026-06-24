# d9_direction1_unsupervised.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")

# ================== 配置 ==================
FEAT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\features")
CAL_FEAT_PATH = FEAT_DIR / "feature_matrix_cal.csv"
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D9_advanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFS_TOP6 = [
    "fusion_vib_energy",
    "fusion_grav_ratio_y",
    "fusion_acc_mag_std",
    "freq_gyro_z_band1_0.5-2Hz_energy_ratio",
    "fusion_pitch_rel_mean",
    "time_basic_acc_y_mean"
]
LABEL_MAP = {0: "静坐", 1: "站立", 2: "行走", 3: "跑步", 4: "上楼", 5: "下楼"}
RANDOM_STATE = 42

# ================== 加载数据 ==================
df = pd.read_csv(CAL_FEAT_PATH)
X = df[SFS_TOP6].values
y = df["label"].values
print(f"✅ 加载标定后特征矩阵：{X.shape[0]} 窗口, {X.shape[1]} 维特征")

# ================== 标准化 ==================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================== 1. K-means 聚类 ==================
print("\n🟢 K-means (K=6) 聚类中...")
kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

km_ari = adjusted_rand_score(y, km_labels)
km_nmi = normalized_mutual_info_score(y, km_labels)
print(f"   K-means  ARI: {km_ari:.4f}  (1=完美, 0=随机)")
print(f"   K-means  NMI: {km_nmi:.4f}  (1=完美, 0=无关)")

# ================== 2. GMM 聚类 ==================
print("\n🟢 GMM (K=6, 全协方差) 聚类中...")
gmm = GaussianMixture(n_components=6, covariance_type='full', random_state=RANDOM_STATE)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)

gmm_ari = adjusted_rand_score(y, gmm_labels)
gmm_nmi = normalized_mutual_info_score(y, gmm_labels)
print(f"   GMM      ARI: {gmm_ari:.4f}")
print(f"   GMM      NMI: {gmm_nmi:.4f}")

# ================== 保存结果 ==================
result_df = pd.DataFrame({
    "真实标签": [LABEL_MAP[i] for i in y],
    "K-means簇": km_labels,
    "GMM簇": gmm_labels
})
result_df.to_csv(OUTPUT_DIR / "无监督聚类结果.csv", index=False, encoding="utf_8_sig")
print(f"\n📁 聚类结果已保存至：{OUTPUT_DIR / '无监督聚类结果.csv'}")

# ================== 交叉表分析 ==================
print("\n========== 各簇与真实标签的对应关系 ==========")
print("K-means 簇 vs 真实标签（前10行）：")
print(pd.crosstab(km_labels, y, rownames=['簇'], colnames=['真实标签']))
print("\nGMM 簇 vs 真实标签（前10行）：")
print(pd.crosstab(gmm_labels, y, rownames=['簇'], colnames=['真实标签']))

# 简单分析
print("\n========== 初步分析 ==========")
if km_ari > 0.5:
    print("✅ K-means 的 ARI 较高，聚类与真实标签有较好的一致性。")
else:
    print("⚠️ K-means 的 ARI 较低，聚类未能准确还原六类活动。")
print("可能原因：静坐/站立特征分布重叠严重，无监督算法难以区分。")

