import streamlit as st
import joblib
import numpy as np

# ════════════════════════════════
# 页面标题
# ════════════════════════════════

st.set_page_config(page_title="污泥预测系统")

st.title("污泥产生量预测系统")

st.write("基于 XGBoost 的气浮与旋流除砂工艺预测")

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

    D10 = st.number_input("D10(μm)", value=10.0)

    D50 = st.number_input("D50(μm)", value=45.0)

    D90 = st.number_input("D90(μm)", value=90.0)

    width = st.number_input("粒径分布宽度", value=2.0)

    ss = st.number_input("进水SS浓度(mg/L)", value=300.0)

    pac = st.number_input("PAC用量(mg/L)", value=100.0)

    ph = st.number_input("pH", value=7.0)

    # ────────────────────────────
    # 预测按钮
    # ────────────────────────────

    if st.button("开始预测"):

        x = np.array([[
            D10,
            D50,
            D90,
            width,
            ss,
            pac,
            ph
        ]])

        pred_log = daf_model.predict(x)

        pred = np.expm1(pred_log)

        st.success(
            f"预测污泥产生量：{pred[0]:.2f} g"
        )

# ════════════════════════════════
# 旋流除砂工艺
# ════════════════════════════════

elif process == "旋流除砂工艺":

    st.subheader("请输入旋流除砂工艺参数")

rock_name = st.selectbox(
    "颗粒岩性",
    [
        "1 - 闪长岩",
        "2 - 花岗岩",
        "3 - 砂岩",
        "4 - 碳质板岩",
        "5 - 泥岩夹砂岩"
    ]
)

rock_map = {
    "1 - 闪长岩": 1,
    "2 - 花岗岩": 2,
    "3 - 砂岩": 3,
    "4 - 碳质板岩": 4,
    "5 - 泥岩夹砂岩": 5
}

rock = rock_map[rock_name]

    density = st.number_input(
        "颗粒密度(g/cm3)",
        value=2.65
    )

    D50 = st.number_input(
        "D50(μm)",
        value=45.0
    )

    width = st.number_input(
        "粒径分布宽度",
        value=2.0
    )

    flow = st.number_input(
        "进水流量(m³/h)",
        value=100.0
    )

    ss = st.number_input(
        "进水SS浓度(mg/L)",
        value=300.0
    )

    # ────────────────────────────
    # 预测按钮
    # ────────────────────────────

    if st.button("开始预测"):

        x = np.array([[
            rock,
            density,
            D50,
            width,
            flow,
            ss
        ]])

        pred_log = cyclone_model.predict(x)

        pred = np.expm1(pred_log)

        st.success(
            f"预测污泥产生量：{pred[0]:.2f} g"
        )
