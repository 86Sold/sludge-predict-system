"""
train_final_models.py
=====================

功能：
1. 训练气浮（DAF）最终 XGBoost 模型
2. 训练旋流除砂（Cyclone）最终 XGBoost 模型
3. 保存模型文件
4. 保存特征顺序文件

输出文件：
--------------------------------
xgb_daf.pkl
xgb_cyclone.pkl

daf_features.json
cyclone_features.json
--------------------------------

说明：
--------------------------------
本代码仅用于最终模型训练与保存

不包含：
× Optuna
× CV
× SHAP
× 绘图
× 多模型对比
--------------------------------
"""

# ════════════════════════════════════════
# 导入库
# ════════════════════════════════════════

import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

# ════════════════════════════════════════
# 数据集配置
# ════════════════════════════════════════

CONFIGS = {

    # ─────────────────────────────
    # 气浮工艺（DAF）
    # ─────────────────────────────
    "daf": {

        "csv": "daf_cleaned.csv",

        "features": [
            "D10(μm)",
            "D50(μm)",
            "D90(μm)",
            "粒径分布宽度",
            "进水SS浓度(mg/L)",
            "混凝剂PAC用量(mg/L)",
            "反应pH值"
        ],

        "target": "污泥产生量(g)",

        "model_save": "xgb_daf.pkl",

        "feature_save": "daf_features.json",

        # 这里替换成你自己的最佳参数
        "params": {
            "n_estimators": 586,
            "learning_rate": 0.056380783210021206,
            "max_depth": 10,
            "subsample": 0.6716372233391855,
            "colsample_bytree": 0.8700455354880448,
            "min_child_weight": 10,
            "reg_alpha": 0.006399770749883031,
            "reg_lambda": 0.0014140553667055112,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        }
    },

    # ─────────────────────────────
    # 旋流除砂工艺（Cyclone）
    # ─────────────────────────────
    "cyclone": {

        "csv": "cyclone_cleaned_reduced.csv",

        "features": [
            "岩性",
            "颗粒密度(g/cm3)",
            "D50(μm)",
            "粒径分布宽度",
            "进水流量(m³/h)",
            "进水SS浓度(mg/L)"
        ],

        "target": "污泥产生量(g)",

        "model_save": "xgb_cyclone.pkl",

        "feature_save": "cyclone_features.json",

        # 这里替换成你自己的最佳参数
        "params": {
            "n_estimators": 454,
            "learning_rate": 0.132080044012035,
            "max_depth": 3,
            "subsample": 0.9479418583132317,
            "colsample_bytree": 0.9855847992840351,
            "min_child_weight": 2,
            "reg_alpha": 0.0004998049254316276,
            "reg_lambda": 0.00014270568576981975,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        }
    }
}

# ════════════════════════════════════════
# 开始训练
# ════════════════════════════════════════

print("=" * 70)
print("最终 XGBoost 模型训练")
print("=" * 70)

for name, cfg in CONFIGS.items():

    print(f"\n正在训练：{name}")

    # ════════════════════════════
    # 读取数据
    # ════════════════════════════

    df = pd.read_csv(cfg["csv"])

    X = df[cfg["features"]].values
    y = df[cfg["target"]].values

    # 与原Step2一致
    y_log = np.log1p(y)

    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")

    # ════════════════════════════
    # 创建模型
    # ════════════════════════════

    model = XGBRegressor(**cfg["params"])

    # ════════════════════════════
    # 训练模型
    # ════════════════════════════

    print("开始训练模型...")

    model.fit(X, y_log)

    print("模型训练完成")

    # ════════════════════════════
    # 保存模型
    # ════════════════════════════

    joblib.dump(model, cfg["model_save"])

    print(f"模型已保存：{cfg['model_save']}")

    # ════════════════════════════
    # 保存特征顺序
    # ════════════════════════════

    with open(cfg["feature_save"], "w", encoding="utf-8") as f:

        json.dump(
            cfg["features"],
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"特征顺序已保存：{cfg['feature_save']}")

    # ════════════════════════════
    # 简单测试
    # ════════════════════════════

    pred_log = model.predict(X[:5])

    pred = np.expm1(pred_log)

    print("\n前5个预测值：")

    for i, p in enumerate(pred):

        print(f"样本{i+1}: {p:.4f} g")

# ════════════════════════════════════════
# 完成
# ════════════════════════════════════════

print("\n" + "=" * 70)
print("全部模型训练完成")
print("=" * 70)

print("\n生成文件：")
print("------------------------------------------------")
print("xgb_daf.pkl")
print("xgb_cyclone.pkl")
print("daf_features.json")
print("cyclone_features.json")
print("------------------------------------------------")