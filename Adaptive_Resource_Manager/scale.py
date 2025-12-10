import joblib

# تحميل الـ scaler
scaler = joblib.load("minmax_scaler.pkl")

# اسماء الـ features
feature_names = ["CPU_Usage","Bandwidth_Usage","Energy_Consumption",
                 "LSTM_Predicted_log","timestamp_numeric","LSTM_timestamp"]

print("Min values:")
for i, f in enumerate(feature_names):
    print(f"{f}: {scaler.data_min_[i]}")

print("\nMax values:")
for i, f in enumerate(feature_names):
    print(f"{f}: {scaler.data_max_[i]}")

# مثال لحساب الـ scaled value يدوي
# x = القيمة الأصلية
# scaled = (x - min) / (max - min)
x_example = 0.5  # ممكن تغيّريها لأي قيمة
min_val = scaler.data_min_[0]  # CPU_Usage
max_val = scaler.data_max_[0]  # CPU_Usage
scaled_example = (x_example - min_val) / (max_val - min_val)
print(f"\nExample scaled value for CPU_Usage={x_example}: {scaled_example}")


