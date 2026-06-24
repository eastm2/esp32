import numpy as np
import pandas as pd
import time
import warnings
from pathlib import Path
from sklearn.model_selection import (
    GroupKFold, GridSearchCV, cross_val_score,
    learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ================== 配置 ==================
FEATURE_PATH = Path(r"E:\1aaa\py\zhuanzhou2026\data\features\feature_matrix_cal.csv")
OUTPUT_DIR = Path(r"E:\1aaa\py\zhuanzhou2026\data\D7_advanced_models")
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
N_SPLITS = 5

# ================== 加载数据 ==================
df = pd.read_csv(FEATURE_PATH)
X = df[SFS_TOP6].values
y = df["label"].values
groups = df["subject_id"].astype(str) + "_" + df["label"].astype(str)
print(f"✅ 加载数据：{X.shape[0]} 样本, {X.shape[1]} 维特征")

# ================== 定义模型与参数网格 ==================
models = {
    "RBF SVM": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.1, 1],
        }
    },
    "kNN": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "params": {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
        }
    },
    "决策树": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__criterion": ["gini", "entropy"],
        }
    },
    "随机森林 (默认权重)": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [1, 3],
            "clf__criterion": ["gini", "entropy"],
        }
    },
    "随机森林 (cost-sensitive)": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                class_weight={0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1},
                random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [1, 3],
            "clf__criterion": ["gini", "entropy"],
        }
    },
    "AdaBoost": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 1.0],
        }
    },
    "梯度提升树": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.01, 0.1],
            "clf__min_samples_leaf": [1, 3],
        }
    },
    "MLP 神经网络": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))
        ]),
        "params": {
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [0.0001, 0.001, 0.01],
            "clf__learning_rate_init": [0.001, 0.01],
        }
    },
}

# ================== 网格搜索 + 记录结果 ==================
cv = GroupKFold(n_splits=N_SPLITS)
results = []

for name, model_info in models.items():
    print(f"\n{'='*60}")
    print(f"调参中：{name}")
    start = time.time()
    gs = GridSearchCV(
        model_info["pipeline"],
        model_info["params"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X, y, groups=groups)
    elapsed = time.time() - start

    # ---------- 保存详细调参记录 ----------
    cv_df = pd.DataFrame(gs.cv_results_)
    cv_df = cv_df.sort_values(by="mean_test_score", ascending=False)
    cv_df["params"] = cv_df["params"].apply(str)
    cv_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
        OUTPUT_DIR / f"tuning_details_{name}.csv",
        index=False,
        encoding="utf_8_sig"
    )

    best_acc = gs.best_score_
    best_params = gs.best_params_
    final_pipeline = gs.best_estimator_
    scores = cross_val_score(
        final_pipeline, X, y, groups=groups,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"最佳参数: {best_params}")
    print(f"最佳交叉验证准确率: {best_acc:.4f}")
    print(f"重评估准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"耗时: {elapsed:.1f}s")

    results.append({
        "模型": name,
        "最佳参数": str(best_params),
        "CV准确率均值": f"{mean_acc:.4f}",
        "CV准确率标准差": f"{std_acc:.4f}",
        "调参耗时(s)": f"{elapsed:.1f}"
    })

# ---------- 汇总所有模型调参记录 ----------
all_tuning = []
for name in models.keys():
    try:
        detail = pd.read_csv(OUTPUT_DIR / f"tuning_details_{name}.csv")
        detail["模型"] = name
        all_tuning.append(detail)
    except FileNotFoundError:
        pass

if all_tuning:
    pd.concat(all_tuning, ignore_index=True).to_csv(
        OUTPUT_DIR / "tuning_records_all_models.csv",
        index=False,
        encoding="utf_8_sig"
    )

# ================== 线性基线对比 ==================
linear_models = {
    "朴素贝叶斯 (GNB)": Pipeline([
        ("scaler", StandardScaler()), ("clf", GaussianNB())
    ]),
    "Fisher LDA": Pipeline([
        ("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())
    ]),
    "逻辑回归": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
}

for name, pipe in linear_models.items():
    scores = cross_val_score(
        pipe, X, y, groups=groups, cv=cv,
        scoring="accuracy", n_jobs=-1
    )
    results.append({
        "模型": name + " (线性基线)",
        "最佳参数": "默认",
        "CV准确率均值": f"{scores.mean():.4f}",
        "CV准确率标准差": f"{scores.std():.4f}",
        "调参耗时(s)": "-"
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "models_comparison.csv",
    index=False,
    encoding="utf_8_sig"
)

# ================== 学习曲线与验证曲线 ==================
print("\n绘制学习曲线与验证曲线...")

# 学习曲线：随机森林过拟合版本
rf_overfit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=100, max_depth=None,
        min_samples_leaf=1, random_state=RANDOM_STATE
    ))
])

train_sizes, train_scores, val_scores = learning_curve(
    rf_overfit, X, y, groups=groups, cv=GroupKFold(n_splits=5),
    train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label="训练集准确率")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, 'o-', label="验证集准确率")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
plt.xlabel("训练样本数")
plt.ylabel("准确率")
plt.title("随机森林学习曲线 (max_depth=None, min_samples_leaf=1)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curve_rf_overfit.png", dpi=300)
plt.close()

# 验证曲线：随机森林 max_depth
param_range = [2, 3, 5, 10, 20, 50, None]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=100, min_samples_leaf=1, random_state=RANDOM_STATE),
    X, y, param_name="max_depth", param_range=param_range,
    groups=groups, cv=GroupKFold(n_splits=5), scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(len(param_range)), train_mean, 'o-', label="训练集准确率")
plt.plot(range(len(param_range)), val_mean, 'o-', label="验证集准确率")
plt.xticks(range(len(param_range)), [str(p) for p in param_range])
plt.xlabel("max_depth")
plt.ylabel("准确率")
plt.title("随机森林 max_depth 验证曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_curve_max_depth.png", dpi=300)
plt.close()

# 验证曲线：随机森林 min_samples_leaf
param_range = [1, 3, 5, 10, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE),
    X, y, param_name="min_samples_leaf", param_range=param_range,
    groups=groups, cv=GroupKFold(n_splits=5), scoring="accuracy", n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(len(param_range)), train_mean, 'o-', label="训练集准确率")
plt.plot(range(len(param_range)), val_mean, 'o-', label="验证集准确率")
plt.xticks(range(len(param_range)), [str(p) for p in param_range])
plt.xlabel("min_samples_leaf")
plt.ylabel("准确率")
plt.title("随机森林 min_samples_leaf 验证曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_curve_min_samples_leaf.png", dpi=300)
plt.close()

# SVM 的 C 参数验证曲线
print("绘制 SVM 的 C 参数验证曲线...")
param_range_C = [0.01, 0.1, 1, 10, 100]
train_scores_svm, val_scores_svm = validation_curve(
    SVC(kernel='rbf', gamma='scale', random_state=RANDOM_STATE),
    StandardScaler().fit_transform(X), y,
    param_name="C", param_range=param_range_C,
    groups=groups, cv=GroupKFold(n_splits=5), scoring="accuracy", n_jobs=-1
)

train_mean_svm = np.mean(train_scores_svm, axis=1)
val_mean_svm = np.mean(val_scores_svm, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(len(param_range_C)), train_mean_svm, 'o-', label="训练集准确率")
plt.plot(range(len(param_range_C)), val_mean_svm, 'o-', label="验证集准确率")
plt.xticks(range(len(param_range_C)), [str(p) for p in param_range_C])
plt.xlabel("C (正则化强度的倒数)")
plt.ylabel("准确率")
plt.title("SVM (RBF) C 参数验证曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "validation_curve_svm_C.png", dpi=300)
plt.close()

# ================== 非线性决策边界可视化 ==================
print("\n绘制非线性决策边界...")

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

boundary_models = {
    "RBF SVM (C=10, gamma=0.1)": SVC(C=10, gamma=0.1, random_state=RANDOM_STATE),
    "随机森林 (max_depth=10)": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    ),
}

for name, clf in boundary_models.items():
    clf.fit(X_lda, y)

    x_min, x_max = X_lda[:, 0].min() - 1, X_lda[:, 0].max() + 1
    y_min, y_max = X_lda[:, 1].min() - 1, X_lda[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10', levels=range(7))
    scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', s=15, alpha=0.8)
    plt.colorbar(scatter, ticks=range(6), label='活动类别')
    plt.title(f'{name} 决策边界 (LDA投影)')
    plt.xlabel('LDA 成分 1')
    plt.ylabel('LDA 成分 2')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'decision_boundary_{name.replace(" ", "_")}.png', dpi=300)
    plt.close()

# ================== 剪枝前后对比打印 ==================
rf_overfit_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, random_state=42)
pipe_overfit = Pipeline([("scaler", StandardScaler()), ("clf", rf_overfit_clf)])
scores_overfit = cross_val_score(pipe_overfit, X, y, groups=groups, cv=GroupKFold(5), n_jobs=-1)

rf_pruned_clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=42)
pipe_pruned = Pipeline([("scaler", StandardScaler()), ("clf", rf_pruned_clf)])
scores_pruned = cross_val_score(pipe_pruned, X, y, groups=groups, cv=GroupKFold(5), n_jobs=-1)

print("\n===== 剪枝效果对比 =====")
print(f"未剪枝验证准确率: {scores_overfit.mean():.4f} ± {scores_overfit.std():.4f}")
print(f"剪枝后验证准确率: {scores_pruned.mean():.4f} ± {scores_pruned.std():.4f}")

print("\n🎉 D7 实验完成！")
print(f"所有结果已保存至：{OUTPUT_DIR}")