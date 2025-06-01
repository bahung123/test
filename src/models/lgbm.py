import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Đọc dữ liệu
train_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/train.csv")
test_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/test.csv")
val_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/val.csv")

# Split features and target
X_train = train_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_train = train_df['isFraud']
X_test = test_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_test = test_df['isFraud']
X_val = val_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_val = val_df['isFraud']

# Scale dữ liệu gốc
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

time_start = time.time()
# Khởi tạo mô hình LightGBM với các tham số tối ưu
model = LGBMClassifier(
    learning_rate=0.1,
    max_depth=4,
    n_estimators=1000,      
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=100,
    early_stopping_rounds=10, 
    reg_alpha=0.25,
    reg_lambda=0.25,
    class_weight='balanced',
    num_leaves=20
)

# Setup evaluation set
eval_set = [(X_train, y_train), (X_val, y_val)]
eval_names = ['train', 'val']

# Huấn luyện mô hình
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    eval_names=eval_names,
    eval_metric=['auc', 'average_precision'], 
    categorical_feature='auto'                 
)

time_end = time.time()
print(f"Thời gian huấn luyện mô hình: {time_end - time_start:.2f} giây")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC: {roc_auc:.4f}")

# AUCPR
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
aucpr = auc(recall, precision)
print(f"\nAUCPR: {aucpr:.4f}")

# Vẽ feature importance
importance = model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importance)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importance[indices], align='center')
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("/home/hung/fraud_detection/src/images/lightgbm/feature_importance.png")
plt.close()

# Tạo DataFrame chứa xác suất và nhãn thật
df_gain = pd.DataFrame({
    'prob': y_prob,
    'label': y_test
})

# Sắp xếp theo xác suất giảm dần
df_gain = df_gain.sort_values(by='prob', ascending=False).reset_index(drop=True)

# Thêm cột decile (chia thành 10 nhóm bằng nhau)
df_gain['decile'] = pd.qcut(df_gain.index, 10, labels=False) + 1

# Tính tổng số outlier thực sự
total_positives = df_gain['label'].sum()

# Tính gain table
gain_table = df_gain.groupby('decile').agg(
    total_samples=('label', 'count'),
    positive_in_decile=('label', 'sum')
).reset_index()

# Cộng dồn số lượng outlier phát hiện được
gain_table['cumulative_positives'] = gain_table['positive_in_decile'].cumsum()

# Tính gain theo phần trăm
gain_table['gain'] = gain_table['cumulative_positives'] / total_positives

# Vẽ biểu đồ gain
plt.figure(figsize=(8, 6))
sns.lineplot(data=gain_table, x='decile', y='gain', marker='o', label='Gain Curve')


plt.plot([1, 10], [0, 1], linestyle='--', color='gray', label='Baseline (random)')

plt.title('Gain Chart on Validation Set')
plt.xlabel('Decile (10% increments)')
plt.ylabel('Cumulative Gain (Recall)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.legend()
plt.savefig("/home/hung/fraud_detection/src/images/lightgbm/gain_chart.png")
plt.close()

# Tạo DataFrame chứa prob và label
df_gain = pd.DataFrame({'prob': y_prob, 'label': y_test})

# Các ngưỡng cutoff
thresholds = np.arange(0, 1.01, 0.05)

results = []
total_positives = df_gain['label'].sum()
X_train = X_train_scaled
for t in thresholds:
    selected = df_gain[df_gain['prob'] >= t]
    positives_in_selected = selected['label'].sum()
    gain = positives_in_selected / total_positives
    results.append({'threshold': t, 'samples_selected': len(selected), 'positives_in_selected': positives_in_selected, 'gain': gain})

df_results = pd.DataFrame(results)

# Vẽ Gain Chart theo ngưỡng
plt.figure(figsize=(8,5))
plt.plot(df_results['threshold'], df_results['gain'], marker='o')
plt.title('Gain Chart theo ngưỡng cutoff')
plt.xlabel('Ngưỡng xác suất (threshold)')
plt.ylabel('Gain (tỉ lệ outlier bắt được)')
plt.grid(True)
plt.savefig("/home/hung/fraud_detection/src/images/lightgbm/gain_chart_threshold.png")
plt.close()

# confusion với ngưỡng 0.925
threshold = 0.925
y_pred_threshold = (y_prob >= threshold).astype(int)
print("\nConfusion Matrix with threshold 0.925:\n", confusion_matrix(y_test, y_pred_threshold))
# Classification report với ngưỡng 0.925
print("\nClassification Report with threshold 0.925:\n", classification_report(y_test, y_pred_threshold))

