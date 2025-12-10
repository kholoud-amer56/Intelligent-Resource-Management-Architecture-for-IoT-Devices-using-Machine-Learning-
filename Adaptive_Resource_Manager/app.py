# app_resource_manager_commented.py
# =========================================
# Adaptive Resource Manager — Professional Decision Engine
# =========================================

# =========================================
# IMPORT LIBRARIES
# =========================================
import streamlit as st       # واجهة المستخدم التفاعلية
import pandas as pd          # التعامل مع البيانات على شكل DataFrame
import numpy as np           # العمليات الحسابية والمصفوفات
import joblib                # لتحميل الموديل المحفوظ والـ scaler
import json                  # لقراءة feature columns
import os                    # التعامل مع الملفات
from datetime import datetime # لتسجيل وقت القرارات
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns        # لرسم heatmaps وVisualization

# =========================================
# STREAMLIT CONFIG
# =========================================
# set_page_config يجب أن يكون أول أمر في الـ Streamlit
st.set_page_config(
    page_title="Adaptive Resource Manager",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# CONFIG FILES
# =========================================
MODEL_PATH = "best_logistic_regression_model.pkl"  # ملف موديل اللوجستيك
SCALER_PATH = "minmax_scaler.pkl"                  # ملف MinMaxScaler
AUDIT_LOG_PATH = "resource_actions_log.csv"       # ملف تسجيل القرارات
TEST_DATA_PATH = "processed_dataset_fixed.xlsx"   # بيانات اختبار اختيارية
FEATURES_PATH = "feature_columns.json"            # أسماء الـ Features المستخدمة في الموديل

# =========================================
# LOAD MODEL, SCALER, FEATURES
# =========================================
# محاولة تحميل الموديل
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")

# تحميل الـ scaler (MinMaxScaler)
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading scaler: {e}")

# تحميل قائمة الـ Features (الأعمدة) المطلوبة من الموديل
try:
    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)
except Exception as e:
    st.error(f"Error loading feature columns: {e}")

# =========================================
# UTILITY FUNCTIONS
# =========================================
def preprocess_row(row: dict):
    """
    تحويل Dictionary إلى DataFrame وحشو القيم المفقودة.
    row: {'CPU_Usage': 0.5, ...}
    """
    df = pd.DataFrame([row], columns=FEATURES)
    df = df.fillna(0)  # استبدال أي NaN بـ 0
    return df

def decide_action_from_probs(prob_dict, thresholds):
    """
    تحديد الإجراء (pause/adjust/offload) بناءً على الاحتمالات والـ thresholds.
    الأولوية: offload > adjust > pause
    """
    if prob_dict.get("offload",0) >= thresholds["offload"]:
        return "offload", prob_dict["offload"], "prob_offload>=threshold"
    if prob_dict.get("adjust",0) >= thresholds["adjust"]:
        return "adjust", prob_dict["adjust"], "prob_adjust>=threshold"
    if prob_dict.get("pause",0) >= thresholds["pause"]:
        return "pause", prob_dict["pause"], "prob_pause>=threshold"
    # fallback: أعلى احتمال إذا لم يصل أي احتمال للـ threshold
    pred = max(prob_dict, key=prob_dict.get)
    return pred, prob_dict[pred], "fallback_highest_prob"

def log_action(record: dict):
    """
    تسجيل القرار في ملف CSV.
    record: {'timestamp':..., 'device_id':..., 'input':..., 'decision':..., ...}
    """
    df = pd.DataFrame([record])
    try:
        if os.path.exists(AUDIT_LOG_PATH):
            # Append without header
            df.to_csv(AUDIT_LOG_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(AUDIT_LOG_PATH, index=False)
    except Exception as e:
        st.error(f"Error logging action: {e}")

# =========================================
# SIDEBAR CONFIGURATION
# =========================================
st.sidebar.header("Configuration")

# إعداد sliders للـ thresholds لكل إجراء
th_offload = st.sidebar.slider("Threshold offload", 0.01, 0.99, 0.60, 0.01)
th_adjust = st.sidebar.slider("Threshold adjust", 0.01, 0.99, 0.50, 0.01)
th_pause = st.sidebar.slider("Threshold pause", 0.01, 0.99, 0.70, 0.01)
thresholds = {"offload": th_offload, "adjust": th_adjust, "pause": th_pause}

st.sidebar.markdown("---")
st.sidebar.write("Audit log file:")
st.sidebar.code(AUDIT_LOG_PATH)
st.sidebar.markdown("---")

# =========================================
# PAGE HEADER
# =========================================
st.title("🔧 Resource Manager — Decision Engine")
st.markdown("Use the interface to predict actions (pause/adjust/offload) and log all decisions.")

# =========================================
# TABS FOR DIFFERENT FUNCTIONALITY
# =========================================
tab1, tab2, tab3 = st.tabs(["Single Inference", "Batch Inference", "Monitoring & Reports"])

# -----------------------------------------
# SINGLE INFERENCE TAB
# -----------------------------------------
with tab1:
    st.header("Single Inference / Decision")
    col1, col2 = st.columns(2)

    # إدخال القيم الفردية لكل Feature
    with col1:
        CPU_Usage = st.number_input("CPU_Usage", 0.0, 1.0, 0.5, format="%.4f")
        Bandwidth_Usage = st.number_input("Bandwidth_Usage", 0.0, 1.0, 0.5, format="%.4f")
        Energy_Consumption = st.number_input("Energy_Consumption", 0.0, 1.0, 0.5, format="%.4f")
    with col2:
        LSTM_Predicted_log = st.number_input("LSTM_Predicted_log", -10.0, 10.0, 0.0, format="%.6f")
        timestamp_numeric = st.number_input("timestamp_numeric", 0.0, 1.0, 0.5, format="%.6f")
        LSTM_timestamp = st.number_input("LSTM_timestamp", 0.0, 1.0, 0.5, format="%.6f")

    device_id = st.text_input("Device ID (optional)", value="device_001")

    if st.button("🔍 Predict & Decide"):
        # إنشاء DataFrame من القيم المدخلة
        sample = {
            "CPU_Usage": CPU_Usage,
            "Bandwidth_Usage": Bandwidth_Usage,
            "Energy_Consumption": Energy_Consumption,
            "LSTM_Predicted_log": LSTM_Predicted_log,
            "timestamp_numeric": timestamp_numeric,
            "LSTM_timestamp": LSTM_timestamp
        }
        df_sample = preprocess_row(sample)

        # تطبيق Min-Max Scaling على البيانات المدخلة
        X_scaled = scaler.transform(df_sample)

        # التنبؤ بالاحتمالات لكل Class
        probs = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probs))

        # اتخاذ القرار النهائي بناءً على thresholds
        action, conf, reason = decide_action_from_probs(prob_dict, thresholds)

        # عرض النتائج على الـ UI
        st.subheader("Result")
        st.write("Model prediction:", classes[np.argmax(probs)])
        st.write("Decision (engine):", action)
        st.write("Confidence:", round(conf, 4))
        st.write("Reason:", reason)
        st.json(prob_dict)

        # تسجيل القرار في ملف Audit log
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "device_id": device_id,
            "input": json.dumps(sample),
            "model_pred": classes[np.argmax(probs)],
            "decision": action,
            "confidence": float(conf),
            "reason": reason
        }
        log_action(record)
        st.success("✅ Decision logged")

# -----------------------------------------
# BATCH INFERENCE TAB
# -----------------------------------------
with tab2:
    st.header("Batch Inference (CSV / Excel)")
    uploaded = st.file_uploader("Upload CSV / XLSX file", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            # قراءة الملف
            if uploaded.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded)
            else:
                batch_df = pd.read_excel(uploaded)

            # التأكد من وجود جميع Features
            missing = set(FEATURES) - set(batch_df.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                batch_df = batch_df[FEATURES].fillna(0)

                # تطبيق Min-Max Scaling
                X_scaled = scaler.transform(batch_df)

                # التنبؤ بالاحتمالات
                probs = model.predict_proba(X_scaled)
                preds = model.predict(X_scaled)
                result_df = batch_df.copy()

                # إضافة الأعمدة الاحتمالية لكل Class
                for i, cls in enumerate(model.classes_):
                    result_df[f"Prob_{cls}"] = probs[:, i]
                result_df["Model_Pred"] = preds

                # اتخاذ القرارات النهائية لكل صف
                decisions = []
                for idx, row in result_df.iterrows():
                    prob_dict = {cls: row[f"Prob_{cls}"] for cls in model.classes_}
                    action, conf, reason = decide_action_from_probs(prob_dict, thresholds)
                    decisions.append((action, conf, reason))

                result_df["Decision"] = [d[0] for d in decisions]
                result_df["Decision_conf"] = [d[1] for d in decisions]
                result_df["Decision_reason"] = [d[2] for d in decisions]

                st.dataframe(result_df.head(200))
                # تنزيل النتائج
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results CSV", data=csv, file_name="batch_predictions_results.csv")

        except Exception as e:
            st.error(f"Error loading file: {e}")

# -----------------------------------------
# MONITORING & REPORTS TAB
# -----------------------------------------
with tab3:
    st.header("Monitoring & Reports")

    # عرض Audit log
    if os.path.exists(AUDIT_LOG_PATH):
        audit_df = pd.read_csv(AUDIT_LOG_PATH)
        st.write("Total logged actions:", len(audit_df))
        st.dataframe(audit_df.tail(100))
    else:
        st.info("No audit log found yet.")

    # عرض الـ min و max لكل Feature من بيانات الاختبار
    if TEST_DATA_PATH and os.path.exists(TEST_DATA_PATH):
        test_df = pd.read_excel(TEST_DATA_PATH)
        df_features = test_df[FEATURES].fillna(0)
        st.subheader("Feature-wise min/max (for manual scaling)")
        st.dataframe(pd.DataFrame({
            "min": df_features.min(),
            "max": df_features.max()
        }))

        # تقييم الموديل على Test Data
        st.subheader("Run Evaluation on Test Data")
        if st.button("Run full evaluation"):
            test_df = test_df.dropna().reset_index(drop=True)
            if not set(FEATURES).issubset(test_df.columns) or "Action_Label" not in test_df.columns:
                st.error("Test file must include features + Action_Label column")
            else:
                X_test = scaler.transform(test_df[FEATURES])
                y_true = test_df["Action_Label"]
                y_pred = model.predict(X_test)
                st.text(classification_report(y_true, y_pred))

                cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
                cm_df = pd.DataFrame(cm, index=[f"Actual:{c}" for c in model.classes_],
                                     columns=[f"Pred:{c}" for c in model.classes_])
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_df, annot=True, fmt="d", cmap='Blues', ax=ax)
                st.pyplot(fig)
                st.success("Evaluation complete")

    # عرض الـ coefficients الخاصة بموديل Logistic Regression
    st.subheader("Model coefficients (Logistic Regression)")
    try:
        if hasattr(model, "coef_"):
            coef_df = pd.DataFrame(model.coef_, columns=FEATURES, index=model.classes_)
            st.dataframe(coef_df.T)
        else:
            st.info("Model has no coefficients to show")
    except Exception as e:
        st.error(f"Could not show coefficients: {e}")

st.markdown("---")
st.caption("Adaptive Resource Manager — Professional Decision Engine with full logging & monitoring")
