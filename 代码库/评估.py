# d8_integrated_final.py —— D8 完整评估（多模型 + 最佳模型深度分析）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score, LeaveOneGroupOut,
    cross_val_predict, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.utils import resample
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置 ==================
FEAT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\features")
RAW_FEAT_PATH = FEAT_DIR / "feature_matrix_raw.csv"
CAL_FEAT_PATH = FEAT_DIR / "feature_matrix_cal.csv"
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D8_Final_Report")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFS_TOP6 = [
    "fusion_vib_energy", "fusion_grav_ratio_y", "fusion_acc_mag_std",
    "freq_gyro_z_band1_0.5-2Hz_energy_ratio", "fusion_pitch_rel_mean", "time_basic_acc_y_mean"
]
LABEL_MAP = {0: "静坐", 1: "站立", 2: "行走", 3: "跑步", 4: "上楼", 5: "下楼"}
RANDOM_STATE = 42
BOOTSTRAP_ITERS = 2000

# ================== 加载数据 ==================
df_cal = pd.read_csv(CAL_FEAT_PATH)
available_features = [f for f in SFS_TOP6 if f in df_cal.columns]
X_cal = df_cal[available_features].values
y_cal = df_cal["label"].values
subject_ids_cal = df_cal["subject_id"].values
print(f"✅ 标定后数据：{X_cal.shape[0]}窗口，{X_cal.shape[1]}维特征，{len(np.unique(subject_ids_cal))}名被试")

# ================== 定义所有参与对比的模型 ==================
def get_models():
    return {
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=RANDOM_STATE))
        ]),
        "Fisher LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis())
        ]),
        "朴素贝叶斯 (GNB)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        "kNN (k=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=3, weights="distance"))
        ]),
        "决策树 (depth=10)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, random_state=RANDOM_STATE))
        ]),
        "随机森林 (默认)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=RANDOM_STATE))
        ]),
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "线性 SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1, probability=True, random_state=RANDOM_STATE))
        ]),
        "高斯贝叶斯 (QDA)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1))
        ]),
    }

# ================== 1. 多模型 LOSO 评估（生成混淆矩阵、指标、汇总表） ==================
logo = LeaveOneGroupOut()
models = get_models()
summary_rows = []
all_preds = {}     # 保存预测结果用于后续检验

for name, pipe in models.items():
    print(f"\n===== {name} =====")
    y_pred = cross_val_predict(pipe, X_cal, y_cal, cv=logo, groups=subject_ids_cal, n_jobs=-1)
    all_preds[name] = y_pred

    acc = accuracy_score(y_cal, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_cal, y_pred, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_cal, y_pred, average="micro", zero_division=0)

    # 各类别指标 CSV
    prec, rec, f1, support = precision_recall_fscore_support(y_cal, y_pred, zero_division=0)
    class_metrics = [{
        "类别": LABEL_MAP[i],
        "Precision": f"{prec[i]:.4f}",
        "Recall": f"{rec[i]:.4f}",
        "F1-score": f"{f1[i]:.4f}",
        "样本数": support[i]
    } for i in range(6)]
    pd.DataFrame(class_metrics).to_csv(OUTPUT_DIR / f"各类别指标_{name}.csv", index=False, encoding="utf_8_sig")

    # 宏/微平均 CSV
    pd.DataFrame({
        "指标": ["宏平均", "微平均"],
        "Precision": [f"{prec_macro:.4f}", f"{prec_micro:.4f}"],
        "Recall": [f"{rec_macro:.4f}", f"{rec_micro:.4f}"],
        "F1-score": [f"{f1_macro:.4f}", f"{f1_micro:.4f}"]
    }).to_csv(OUTPUT_DIR / f"宏微平均_{name}.csv", index=False, encoding="utf_8_sig")

    # 混淆矩阵图
    cm = confusion_matrix(y_cal, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(LABEL_MAP.values()))
    fig, ax = plt.subplots(figsize=(10,8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"LOSO 归一化混淆矩阵 ({name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"混淆矩阵_{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # AUC（能输出概率的模型）
    auc_val = "N/A"
    if hasattr(pipe, 'predict_proba'):
        pipe.fit(X_cal, y_cal)   # 全量拟合用于获取概率（仅画图）
        y_prob = pipe.predict_proba(X_cal)
        auc_val = f"{roc_auc_score(y_cal, y_prob, multi_class='ovr', average='macro'):.4f}"

    summary_rows.append({
        "模型": name,
        "LOSO准确率": f"{acc:.4f}",
        "宏平均F1": f"{f1_macro:.4f}",
        "微平均F1": f"{f1_micro:.4f}",
        "宏平均Recall": f"{rec_macro:.4f}",
        "宏平均Precision": f"{prec_macro:.4f}",
        "AUC": auc_val
    })
    print(f"  准确率={acc:.4f}, 宏F1={f1_macro:.4f}")

# 汇总表
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "多模型_LOSO_全面对比.csv", index=False, encoding="utf_8_sig")
print("\n📊 多模型 LOSO 全面对比表已保存")

# 多模型 AUC 柱状图（仅概率模型）
prob_models = summary_df[summary_df["AUC"] != "N/A"]
if len(prob_models) > 0:
    auc_values = [float(v) for v in prob_models["AUC"]]
    plt.figure(figsize=(10,6))
    plt.barh(prob_models["模型"], auc_values, color='steelblue')
    plt.xlabel('宏平均 AUC')
    plt.title('多模型 LOSO 宏平均 AUC 对比')
    plt.xlim(0.9, 1.0)
    for bar, auc in zip(plt.barh(prob_models["模型"], auc_values), auc_values):
        plt.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{auc:.4f}', va='center')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "多模型_AUC对比.png", dpi=300)
    plt.close()

# ================== 2. 评估协议对比（RBF SVM） ==================
print("\n========== 评估协议对比 ==========")
pipe_svm = models["RBF SVM"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores_kfold = cross_val_score(pipe_svm, X_cal, y_cal, cv=skf, scoring="accuracy", n_jobs=-1)
scores_loso = cross_val_score(pipe_svm, X_cal, y_cal, cv=logo, groups=subject_ids_cal, scoring="accuracy", n_jobs=-1)
print(f"随机5折准确率：{scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")
print(f"LOSO准确率：    {scores_loso.mean():.4f} ± {scores_loso.std():.4f}")
print(f"乐观偏差：{(scores_kfold.mean()-scores_loso.mean())*100:.1f} 个百分点")

# ================== 3. 最佳模型 (RBF SVM) 详细可视化 ==================
print("\n========== 最佳模型 (RBF SVM) 详细评估 ==========")
best_pipe = models["RBF SVM"]
best_pipe.fit(X_cal, y_cal)
y_prob = best_pipe.predict_proba(X_cal)

# ROC 曲线
roc_auc = roc_auc_score(y_cal, y_prob, multi_class="ovr", average="macro")
fig, ax = plt.subplots(figsize=(10,8))
for i in range(6):
    y_true_binary = (y_cal == i).astype(int)
    y_score = y_prob[:, i]
    RocCurveDisplay.from_predictions(y_true_binary, y_score, name=LABEL_MAP[i], ax=ax)
plt.plot([0,1],[0,1], "k--", alpha=0.5)
plt.title(f"多分类 ROC 曲线 - 宏平均 AUC = {roc_auc:.4f}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "最佳模型_ROC.png", dpi=300)
plt.close()
print(f"✅ ROC 曲线已保存，AUC = {roc_auc:.4f}")

# PR 曲线
fig, ax = plt.subplots(figsize=(10,8))
for i in range(6):
    y_true_binary = (y_cal == i).astype(int)
    y_score = y_prob[:, i]
    PrecisionRecallDisplay.from_predictions(y_true_binary, y_score, name=LABEL_MAP[i], ax=ax)
plt.title("多分类 PR 曲线 (RBF SVM)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "最佳模型_PR.png", dpi=300)
plt.close()
print("✅ PR 曲线已保存")

# ================== 4. Bootstrap 置信区间 ==================
print("\n========== 置信区间 (Bootstrap) ==========")
y_pred_svm = all_preds["RBF SVM"]
n = len(y_cal)
boot_accs = [accuracy_score(y_cal[resample(range(n), replace=True)], y_pred_svm[resample(range(n), replace=True)])
             for _ in range(BOOTSTRAP_ITERS)]
ci_low = np.percentile(boot_accs, 2.5)
ci_high = np.percentile(boot_accs, 97.5)
mean_acc = np.mean(boot_accs)
print(f"准确率 95% CI: [{ci_low:.4f}, {ci_high:.4f}] (均值 {mean_acc:.4f})")

# ================== 5. McNemar 显著性检验（前三名） ==================
print("\n========== 前三名 McNemar 检验 ==========")
top3 = ["RBF SVM", "朴素贝叶斯 (GNB)", "Fisher LDA"]
for i in range(len(top3)):
    for j in range(i+1, len(top3)):
        a, b = top3[i], top3[j]
        mask = y_cal == 1  # 站立类
        correct_a = (all_preds[a][mask] == y_cal[mask]).astype(int)
        correct_b = (all_preds[b][mask] == y_cal[mask]).astype(int)
        n11 = np.sum((correct_a == 1) & (correct_b == 1))
        n10 = np.sum((correct_a == 1) & (correct_b == 0))
        n01 = np.sum((correct_a == 0) & (correct_b == 1))
        n00 = np.sum((correct_a == 0) & (correct_b == 0))
        pval = mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue
        print(f"{a} vs {b}: p = {pval:.6f} {'**' if pval < 0.05 else ''}")

# ================== 6. 标定增益对比 ==================
print("\n========== 标定增益 ==========")
if RAW_FEAT_PATH.exists():
    df_raw = pd.read_csv(RAW_FEAT_PATH)
    X_raw = df_raw[[f for f in SFS_TOP6 if f in df_raw.columns]].values
    y_raw = df_raw["label"].values
    subject_ids_raw = df_raw["subject_id"].values
    raw_scores = cross_val_score(pipe_svm, X_raw, y_raw, cv=logo, groups=subject_ids_raw, scoring="accuracy", n_jobs=-1)
    cal_scores = cross_val_score(pipe_svm, X_cal, y_cal, cv=logo, groups=subject_ids_cal, scoring="accuracy", n_jobs=-1)
    gain = (cal_scores.mean() - raw_scores.mean()) * 100
    print(f"标定前: {raw_scores.mean():.4f} ± {raw_scores.std():.4f}")
    print(f"标定后: {cal_scores.mean():.4f} ± {cal_scores.std():.4f}")
    print(f"增益: +{gain:.1f} 个百分点")

    plt.figure(figsize=(6,6))
    plt.bar(["标定前","标定后"], [raw_scores.mean(), cal_scores.mean()],
            yerr=[raw_scores.std(), cal_scores.std()], color=["gray","steelblue"], capsize=10)
    plt.ylabel("LOSO准确率")
    plt.title(f"标定增益: +{gain:.1f} 个百分点")
    for i, s in enumerate([raw_scores.mean(), cal_scores.mean()]):
        plt.text(i, s+0.01, f"{s:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "标定增益.png", dpi=300)
    plt.close()
else:
    print("⚠️ 标定前特征矩阵不存在")

# ================== 7. 泛化间隙 ==================
print("\n========== 泛化间隙 ==========")
train_accs, test_accs = [], []
for train_idx, test_idx in logo.split(X_cal, y_cal, groups=subject_ids_cal):
    X_train, X_test = X_cal[train_idx], X_cal[test_idx]
    y_train, y_test = y_cal[train_idx], y_cal[test_idx]
    pipe_svm.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, pipe_svm.predict(X_train)))
    test_accs.append(accuracy_score(y_test, pipe_svm.predict(X_test)))
gap = np.mean(train_accs) - np.mean(test_accs)
print(f"训练-测试: {gap:.4f} ({'轻微过拟合' if gap > 0.1 else '泛化良好'})")

print("\n🎉 D8 完整评估结束！所有结果已保存至", OUTPUT_DIR)