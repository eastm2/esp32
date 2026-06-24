# d6_integrated.py —— D6 分类器基线实验 + 手写高斯贝叶斯验证
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import time

plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置 ==================
FEATURE_PATH = Path(r"E:\1aaa\py\zhuanzhou2026\data\features\feature_matrix_cal.csv")
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D6_baseline_classifiers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFS_TOP6_FEATURES = [
    "fusion_vib_energy", "fusion_grav_ratio_y", "fusion_acc_mag_std",
    "freq_gyro_z_band1_0.5-2Hz_energy_ratio", "fusion_pitch_rel_mean", "time_basic_acc_y_mean"
]

LABEL_MAP = {0: "静坐", 1: "站立", 2: "行走", 3: "跑步", 4: "上楼", 5: "下楼"}
RANDOM_STATE = 42
CV_SPLITS = 5

# ================== 加载数据 ==================
df = pd.read_csv(FEATURE_PATH)
X = df[SFS_TOP6_FEATURES].values
y = df["label"].values
groups = df["subject_id"].astype(str) + "_" + df["label"].astype(str)

print(f"✅ 加载数据：{X.shape[0]} 样本, {X.shape[1]} 维特征")

# ================== 手写高斯贝叶斯分类器 ==================
class GaussianBayesClassifier:
    """手写高斯贝叶斯分类器（QDA），支持协方差正则化"""
    def __init__(self, reg_param=0.0):
        self.reg_param = reg_param
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covariances_ = np.zeros((n_classes, n_features, n_features))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.priors_[i] = len(X_cls) / len(X)
            self.means_[i] = np.mean(X_cls, axis=0)
            cov = np.cov(X_cls, rowvar=False)
            if self.reg_param > 0:
                cov += np.eye(n_features) * self.reg_param
            self.covariances_[i] = cov
        return self

    def _log_likelihood(self, X, class_idx):
        n_features = X.shape[1]
        mean = self.means_[class_idx]
        cov = self.covariances_[class_idx]
        try:
            cov_inv = np.linalg.inv(cov)
            _, log_det = np.linalg.slogdet(cov)
        except np.linalg.LinAlgError:
            raise ValueError(f"类 {self.classes_[class_idx]} 协方差矩阵奇异，请增大 reg_param")
        centered = X - mean
        mahalanobis = np.sum(np.dot(centered, cov_inv) * centered, axis=1)
        log_likelihood = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        return log_likelihood

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            log_probs[:, i] = self._log_likelihood(X, i) + np.log(self.priors_[i])
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

# ================== 1. 基础分类器训练与评估 ==================
classifiers = {
    "高斯贝叶斯 (QDA)": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "朴素贝叶斯 (GNB)": GaussianNB(),
    "Fisher LDA": LinearDiscriminantAnalysis(),
    "逻辑回归 (LR)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "线性 SVM": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
}

results = []
for name, clf in classifiers.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv = GroupKFold(n_splits=CV_SPLITS)
    scores = cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    results.append({"分类器": name, "准确率均值": f"{scores.mean():.4f}", "准确率标准差": f"{scores.std():.4f}"})

baseline_df = pd.DataFrame(results)
baseline_df.to_csv(OUTPUT_DIR / "baseline_accuracy.csv", index=False, encoding="utf_8_sig")

# ================== 2. 最小风险贝叶斯决策（使用 sklearn QDA） ==================
num_classes = 6
cost_matrix = np.ones((num_classes, num_classes))
np.fill_diagonal(cost_matrix, 0)
cost_matrix[2][5] = 5  # 真实下楼 → 行走
cost_matrix[4][5] = 3  # 真实下楼 → 上楼
cost_matrix[0][1] = 3  # 真实站立 → 静坐

cv = GroupKFold(n_splits=CV_SPLITS)
qda = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1))
])

y_true_all, y_pred_min_error, y_pred_min_risk = [], [], []
for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    qda.fit(X_train, y_train)
    proba = qda.predict_proba(X_test)
    y_pred_err = np.argmax(proba, axis=1)
    risk = proba @ cost_matrix.T
    y_pred_risk = np.argmin(risk, axis=1)
    y_true_all.extend(y_test)
    y_pred_min_error.extend(y_pred_err)
    y_pred_min_risk.extend(y_pred_risk)

acc_error = accuracy_score(y_true_all, y_pred_min_error)
acc_risk = accuracy_score(y_true_all, y_pred_min_risk)
cm_error = confusion_matrix(y_true_all, y_pred_min_error, labels=range(6))
cm_risk = confusion_matrix(y_true_all, y_pred_min_risk, labels=range(6))
recalls_error = recall_score(y_true_all, y_pred_min_error, average=None)
recalls_risk = recall_score(y_true_all, y_pred_min_risk, average=None)

print("\n===== 最小风险贝叶斯决策 (sklearn QDA) =====")
print(f"最小错误率准确率: {acc_error:.4f}")
print(f"最小风险准确率:   {acc_risk:.4f}")
print("\n各类别召回率对比：")
for idx, name in LABEL_MAP.items():
    print(f"  {name}: 错误率={recalls_error[idx]:.4f} → 风险={recalls_risk[idx]:.4f}")

# 保存混淆矩阵图片
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].matshow(cm_error, cmap='Blues')
axes[0].set_title("最小错误率决策混淆矩阵")
axes[0].set_xlabel("预测标签"); axes[0].set_ylabel("真实标签")
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, cm_error[i, j], ha='center', va='center', fontsize=8)
axes[1].matshow(cm_risk, cmap='Blues')
axes[1].set_title("最小风险决策混淆矩阵")
axes[1].set_xlabel("预测标签"); axes[1].set_ylabel("真实标签")
for i in range(6):
    for j in range(6):
        axes[1].text(j, i, cm_risk[i, j], ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300)
plt.close()

# ================== 3. 决策边界可视化 ==================
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_lda, y)

x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(12, 9))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab10", levels=range(7))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="tab10", s=15, alpha=0.8)
plt.colorbar(scatter, ticks=range(6), label="活动类别")
plt.title("逻辑回归决策边界 (LDA投影空间)")
plt.xlabel("LDA 成分 1")
plt.ylabel("LDA 成分 2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decision_boundary_LR.png", dpi=300)
plt.close()

# ================== 4. 手写高斯贝叶斯与 sklearn 对齐验证 ==================
print("\n===== 手写高斯贝叶斯验证 =====")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sklearn_qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
sklearn_qda.fit(X_scaled, y)
y_pred_sklearn = sklearn_qda.predict(X_scaled)

my_qda = GaussianBayesClassifier(reg_param=0.1)
my_qda.fit(X_scaled, y)
y_pred_my = my_qda.predict(X_scaled)

agreement = np.mean(y_pred_sklearn == y_pred_my)
print(f"手写与 sklearn QDA 预测一致率: {agreement:.4%}")
print(f"sklearn QDA 训练集准确率: {accuracy_score(y, y_pred_sklearn):.4f}")
print(f"手写 QDA 训练集准确率: {accuracy_score(y, y_pred_my):.4f}")

# 如果一致率不是 100%，显示差异样本
diff_idx = np.where(y_pred_sklearn != y_pred_my)[0]
if len(diff_idx) > 0:
    print(f"不一致样本数: {len(diff_idx)}，示例如下：")
    for idx in diff_idx[:10]:
        print(f"样本 {idx}: 真实={y[idx]}, sklearn={y_pred_sklearn[idx]}, 手写={y_pred_my[idx]}")
else:
    print("🎉 完美对齐！手写高斯贝叶斯与 sklearn 实现数学等价。")

print("\n🎉 D6 实验（含手写高斯贝叶斯）完成！")
print(f"结果已保存至：{OUTPUT_DIR}")