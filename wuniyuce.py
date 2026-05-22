import streamlit as st
import joblib
import numpy as np

# ════════════════════════════════
# 页面配置
# ════════════════════════════════

st.set_page_config(
    page_title="污泥预测系统",
    layout="centered"
)

st.title("污泥产生量预测系统")

st.write("基于 XGBoost 的旋流除砂与气浮工艺预测")

# ════════════════════════════════
# 加载模型
# ════════════════════════════════

daf_model = joblib.load("xgb_daf.pkl")

cyclone_model = joblib.load("xgb_cyclone.pkl")

# ════════════════════════════════
# 工艺选择
# ════════════════════════════════

process = st.selectbox(
    "选择工艺",
    ["气浮工艺", "旋流除砂工艺"]
)

# ════════════════════════════════
# 气浮工艺
# ════════════════════════════════

if process == "气浮工艺":

    st.subheader("请输入气浮工艺参数")

    D10 = st.number_input(
        "D10(μm)",
        value=10.0,
        format="%f"
    )

    D50 = st.number_input(
        "D50(μm)",
        value=45.0,
        format="%f"
    )

    D90 = st.number_input(
        "D90(μm)",
        value=90.0,
        format="%f"
    )

    # 自动计算粒径分布宽度
    if D50 != 0:
        width = (D90 - D10) / D50
    else:
        width = 0.0

    st.info(
        f"自动计算粒径分布宽度：{width:.6f}"
    )

    ss = st.number_input(
        "进水SS浓度(mg/L)",
        value=300.0,
        format="%f"
    )

    pac = st.number_input(
        "PAC用量(mg/L)",
        value=100.0,
        format="%f"
    )

    ph = st.number_input(
        "pH",
        value=7.0,
        format="%f"
    )

    # ────────────────────────────
    # 预测按钮
    # ────────────────────────────

    if st.button("开始预测"):

        x = np.array([[
            float(D10),
            float(D50),
            float(D90),
            float(width),
            float(ss),
            float(pac),
            float(ph)
        ]], dtype=np.float32)

        pred_log = float(daf_model.predict(x)[0])

        pred = float(np.expm1(pred_log))

        st.success(
            f"预测污泥产生量：{pred:.4f} g"
        )

# ════════════════════════════════
# 旋流除砂工艺
# ════════════════════════════════

elif process == "旋流除砂工艺":

    st.subheader("请输入旋流除砂工艺参数")

    # ────────────────────────────
    # 岩性选择（自动绑定颗粒密度）
    # ────────────────────────────

    rock_options = {
        "1 - 闪长岩": {
            "code": 1,
            "density": 2.6502
        },
        "2 - 花岗岩": {
            "code": 2,
            "density": 2.6631
        },
        "3 - 砂岩": {
            "code": 3,
            "density": 2.6960
        },
        "4 - 碳质板岩": {
            "code": 4,
            "density": 2.5777
        },
        "5 - 泥岩夹砂岩": {
            "code": 5,
            "density": 2.6936
        }
    }

    rock_name = st.selectbox(
        "颗粒岩性",
        list(rock_options.keys())
    )

    rock = float(
        rock_options[rock_name]["code"]
    )

    density = float(
        rock_options[rock_name]["density"]
    )

    st.info(
        f"自动匹配颗粒密度：{density:.4f} g/cm³"
    )

    D50 = st.number_input(
        "D50(μm)",
        value=45.0,
        format="%f"
    )

    width = st.number_input(
        "粒径分布宽度",
        value=2.0,
        format="%f"
    )

    flow = st.number_input(
        "进水流量(m³/h)",
        value=100.0,
        format="%f"
    )

    ss = st.number_input(
        "进水SS浓度(mg/L)",
        value=300.0,
        format="%f"
    )

    # ────────────────────────────
    # 预测按钮
    # ────────────────────────────

    if st.button("开始预测"):

        x = np.array([[
            float(rock),
            float(density),
            float(D50),
            float(width),
            float(flow),
            float(ss)
        ]], dtype=np.float32)

        pred_log = float(cyclone_model.predict(x)[0])

        pred = float(np.expm1(pred_log))

        st.success(
            f"预测污泥产生量：{pred:.4f} g"
        )
