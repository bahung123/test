import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import time
import seaborn as sns

# Đọc dữ liệu
train_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/train_combined.csv")
test_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/test_combined.csv")
val_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/val_combined.csv")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Tách features và target
X_train = train_df.drop(['isFraud'], axis=1)
y_train = train_df['isFraud']
X_test = test_df.drop(['isFraud'], axis=1)
y_test = test_df['isFraud']
X_val = val_df.drop(['isFraud'], axis=1)
y_val = val_df['isFraud']

# Khởi tạo mô hình XGBoost với best params
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=1000,
    subsample=1,
    colsample_bytree=0.9,
    eval_metric=['auc', 'aucpr'], 
    random_state=42,
    reg_alpha=0.2,
    reg_lambda=1.0,
    early_stopping_rounds=10
)
# Thiết lập eval_set để theo dõi
eval_set = [(X_train, y_train), (X_val, y_val)]
# Huấn luyện mô hình

model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=True
)

# Dự đoán
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# Đánh giá
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", roc_auc)
# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC Score:", pr_auc)
# Vẽ biểu đồ ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('/home/hung/fraud_detection/src/images/xgboost_hidden/roc_curve.png')



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
plt.savefig('/home/hung/fraud_detection/src/images/xgboost_hidden/gain_chart_decile.png')


# Tạo DataFrame chứa prob và label
df_gain = pd.DataFrame({'prob': y_prob, 'label': y_test})

# Các ngưỡng cutoff
thresholds = np.arange(0, 1.01, 0.05)

results = []
total_positives = df_gain['label'].sum()

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
plt.savefig('/home/hung/fraud_detection/src/images/xgboost_hidden/gain_chart.png')



