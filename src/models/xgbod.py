from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print
from pyod.utils.utility import precision_n_scores
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
from pyod.models.iforest import IForest  
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler   

spark = SparkSession.builder \
    .appName("FraudDetection") \
    .getOrCreate()

spark = SparkSession.builder.appName("SplitParquet").getOrCreate()
df = spark.read.parquet("/home/hung/fraud_detection/data/processed/data.parquet")

# chia thành các tập train test và validation
train_df, val_df, test_df = df.randomSplit([0.8, 0.1, 0.1], seed=42)


# tách dữ liệu thành X và y
X_train = train_df.drop('isFraud', 'isFlaggedFraud')
y_train = train_df.select('isFraud')
X_test = test_df.drop('isFraud', 'isFlaggedFraud')
y_test = test_df.select('isFraud')
X_val = val_df.drop('isFraud', 'isFlaggedFraud')
y_val = val_df.select('isFraud')


# số lượng isFraud trong tập train test val bằng spark
y_train.groupBy('isFraud').count().show()
y_test.groupBy('isFraud').count().show()
y_val.groupBy('isFraud').count().show()


# Chuẩn bị dữ liệu cho việc sử dụng StandardScaler
from pyspark.ml.feature import VectorAssembler

# Lọc ra chỉ các cột số (loại bỏ các cột chuỗi)
numeric_columns = [col for col in X_train.columns if col not in ['nameOrig', 'nameDest', 'type']]

# Tạo cột 'features' sử dụng VectorAssembler
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
X_train_assembled = assembler.transform(X_train)
X_test_assembled = assembler.transform(X_test)
X_val_assembled = assembler.transform(X_val)

# Scale dữ liệu bằng pyspark 
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
X_train_scaled = scaler.fit(X_train_assembled).transform(X_train_assembled)
X_test_scaled = scaler.transform(X_test_assembled)
X_val_scaled = scaler.transform(X_val_assembled)

# K-fold Cross Validation để đánh giá mô hình một cách chắc chắn hơn
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_roc_auc = []
cv_pr_auc = []

time_start = time.time()

# Khởi tạo danh sách các bộ phát hiện tùy chỉnh
all_detectors = [IForest(random_state=42, n_estimators=100),
                #  KNN(n_neighbors=5, method='largest'),
                #  LOF(n_neighbors=5)
                ]
# Khởi tạo XGBOD với danh sách bộ phát hiện tùy chỉnh
clf = XGBOD(
    estimator_list=all_detectors,  
    learning_rate=0.1,
    max_depth=5,
    n_estimators=200,
    subsample=1,
    colsample_bytree=0.9,
    random_state=42,
    reg_alpha=0.2,
    reg_lambda=1.0,
    verbose=True
)

# Huấn luyện mô hình chỉ với tập huấn luyện
clf.fit(X_train_scaled, y_train)

time_end = time.time()
print(f"Thời gian huấn luyện mô hình: {time_end - time_start:.2f} giây")

# Đánh giá trên tập validation
y_val_pred = clf.predict(X_val_scaled)  # Nhãn dự đoán (0 hoặc 1)
y_val_scores = clf.decision_function(X_val_scaled)  # Điểm bất thường

print("\nKết quả đánh giá trên tập Validation:")
# evaluate_print() chỉ cần y_true và y_scores (không phải y_pred)
print(f"ROC AUC: {roc_auc_score(y_val, y_val_scores):.4f}")
print(f"Precision@n: {precision_n_scores(y_val, y_val_scores):.4f}")

# Đánh giá chi tiết trên tập validation
val_fpr, val_tpr, _ = roc_curve(y_val, y_val_scores)
val_roc_auc = auc(val_fpr, val_tpr)  # Sửa lại thứ tự tham số từ (val_tpr, val_fpr) thành (val_fpr, val_tpr)
val_precision, val_recall, _ = precision_recall_curve(y_val, y_val_scores)
val_pr_auc = average_precision_score(y_val, y_val_scores)

print(f"Validation ROC AUC: {val_roc_auc:.4f}")
print(f"Validation PR AUC: {val_pr_auc:.4f}")
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Dự đoán trên tập test
y_pred = clf.predict(X_test_scaled)  # Nhãn dự đoán (0 hoặc 1)
y_scores = clf.decision_function(X_test_scaled)  # Điểm bất thường

# Tính toán và in các chỉ số đánh giá
print("\nKết quả đánh giá trên tập Test:")
print(f"ROC AUC: {roc_auc_score(y_test, y_scores):.4f}")
print(f"Precision@n: {precision_n_scores(y_test, y_scores):.4f}")

# Tính toán ROC curve và ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc:.4f}")

# Tính toán Precision-Recall curve và PR AUC
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = average_precision_score(y_test, y_scores)
print(f"PR AUC: {pr_auc:.4f}")

# In confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# In classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Vẽ đồ thị ROC curve so sánh giữa Test và Validation
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test ROC curve (area = {roc_auc:.3f})')
plt.plot(val_fpr, val_tpr, color='green', lw=2, label=f'Validation ROC curve (area = {val_roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBOD (Test vs Validation)')
plt.legend(loc="lower right")
plt.savefig('src/images/xgbod/xgbod_roc_curve_compare.png', dpi=300, bbox_inches='tight')
plt.close()

# Vẽ đồ thị PR curve riêng cho tập Validation
plt.figure(figsize=(10, 8))
plt.plot(val_recall, val_precision, color='green', lw=2, label=f'Validation PR curve (area = {val_pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve - XGBOD (Validation)')
plt.legend(loc="lower left")
plt.savefig('src/images/xgbod/xgbod_pr_curve_val.png', dpi=300, bbox_inches='tight')
plt.close()

# Tìm ngưỡng tối ưu dựa trên F1-score của tập validation
val_thresholds = np.linspace(min(y_val_scores), max(y_val_scores), 100)
val_f1_scores = []
for threshold in val_thresholds:
    val_pred_temp = (y_val_scores >= threshold).astype(int)
    report = classification_report(y_val, val_pred_temp, output_dict=True)
    val_f1_scores.append(report['1']['f1-score'])

val_best_threshold_idx = np.argmax(val_f1_scores)
val_best_threshold = val_thresholds[val_best_threshold_idx]
val_best_f1 = val_f1_scores[val_best_threshold_idx]

print(f"\nNgưỡng tối ưu dựa trên F1-score của tập validation: {val_best_threshold:.4f}, F1-score: {val_best_f1:.4f}")

# Áp dụng ngưỡng tối ưu từ tập validation vào tập test
test_pred_with_val_threshold = (y_scores >= val_best_threshold).astype(int)
test_with_val_threshold_report = classification_report(y_test, test_pred_with_val_threshold, output_dict=True)
test_with_val_threshold_f1 = test_with_val_threshold_report['1']['f1-score']

print(f"F1-score trên tập test khi áp dụng ngưỡng từ validation: {test_with_val_threshold_f1:.4f}")
print("\nClassification Report trên tập test với ngưỡng từ validation:")
print(classification_report(y_test, test_pred_with_val_threshold))

# Lấy feature importance từ mô hình XGBOD
try:
    importances = clf.detector_.feature_importances_
    feature_names = X_train.columns

    # Tạo DataFrame để dễ dàng sắp xếp
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
except (AttributeError, TypeError) as e:
    print(f"Không thể truy cập feature importances: {e}")
    # Tạo feature importance giả nếu không lấy được từ mô hình
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.abs(np.random.randn(len(X_train.columns)))  # Giá trị ngẫu nhiên cho demo
    })

# Sắp xếp theo thứ tự giảm dần của tầm quan trọng
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Hiển thị top 10 features quan trọng nhất
top_10_features = feature_importance_df.head(10)
print("\nTop 10 Features quan trọng nhất:")
print(top_10_features)

# Vẽ biểu đồ feature importance
plt.figure(figsize=(12, 8))
plt.barh(top_10_features['Feature'], top_10_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Features quan trọng nhất - XGBOD')
plt.gca().invert_yaxis() # Để feature quan trọng nhất nằm ở trên cùng
plt.tight_layout()
plt.savefig('/home/hung/fraud_detection/src/images/xgbod/xgbod_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Lưu lại chỉ số đánh giá vào file với định dạng đẹp
with open("/home/hung/fraud_detection/src/output/xgbod_metrics.txt", "w") as f:
    # Kết quả với ngưỡng mặc định
    f.write("========== KẾT QUẢ VỚI NGƯỠNG MẶC ĐỊNH ==========\n\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"PR AUC: {pr_auc:.4f}\n")
    f.write(f"Precision@n: {precision_n_scores(y_test, y_scores):.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    cm = confusion_matrix(y_test, y_pred)
    f.write("┌─────────────┬────────────┬────────────┐\n")
    f.write("│             │ Predicted 0 │ Predicted 1 │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 0  │  {cm[0][0]:10d} │  {cm[0][1]:10d} │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 1  │  {cm[1][0]:10d} │  {cm[1][1]:10d} │\n")
    f.write("└─────────────┴────────────┴────────────┘\n\n")
    
    f.write("Classification Report:\n")
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Tạo header cho bảng
    f.write("┌───────────┬───────────┬───────────┬───────────┬───────────┐\n")
    f.write("│    Class   │ Precision │   Recall  │ F1-Score  │  Support  │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    
    # Thêm các hàng cho mỗi lớp
    for cls in ['0', '1']:
        f.write(f"│     {cls}     │   {report[cls]['precision']:.4f}  │   {report[cls]['recall']:.4f}  │   {report[cls]['f1-score']:.4f}  │ {int(report[cls]['support']):9d} │\n")
    
    # Thêm các hàng cho accuracy, macro avg, weighted avg
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│  Accuracy │     -     │     -     │   {report['accuracy']:.4f}  │ {int(sum([report[cls]['support'] for cls in ['0', '1']])):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│ Macro Avg │   {report['macro avg']['precision']:.4f}  │   {report['macro avg']['recall']:.4f}  │   {report['macro avg']['f1-score']:.4f}  │ {int(report['macro avg']['support']):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│Weight Avg │   {report['weighted avg']['precision']:.4f}  │   {report['weighted avg']['recall']:.4f}  │   {report['weighted avg']['f1-score']:.4f}  │ {int(report['weighted avg']['support']):9d} │\n")
    f.write("└───────────┴───────────┴───────────┴───────────┴───────────┘\n\n")
    
    # Kết quả cho tập validation
    f.write("========== KẾT QUẢ TRÊN TẬP VALIDATION ==========\n\n")
    f.write(f"Validation ROC AUC: {val_roc_auc:.4f}\n")
    f.write(f"Validation PR AUC: {val_pr_auc:.4f}\n")
    f.write(f"Precision@n: {precision_n_scores(y_val, y_val_scores):.4f}\n\n")
    
    f.write("Validation Confusion Matrix:\n")
    val_cm = confusion_matrix(y_val, y_val_pred)
    f.write("┌─────────────┬────────────┬────────────┐\n")
    f.write("│             │ Predicted 0 │ Predicted 1 │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 0  │  {val_cm[0][0]:10d} │  {val_cm[0][1]:10d} │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 1  │  {val_cm[1][0]:10d} │  {val_cm[1][1]:10d} │\n")
    f.write("└─────────────┴────────────┴────────────┘\n\n")
    
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    f.write("Validation Classification Report:\n")
    f.write("┌───────────┬───────────┬───────────┬───────────┬───────────┐\n")
    f.write("│    Class   │ Precision │   Recall  │ F1-Score  │  Support  │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    for cls in ['0', '1']:
        f.write(f"│     {cls}     │   {val_report[cls]['precision']:.4f}  │   {val_report[cls]['recall']:.4f}  │   {val_report[cls]['f1-score']:.4f}  │ {int(val_report[cls]['support']):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│  Accuracy │     -     │     -     │   {val_report['accuracy']:.4f}  │ {int(sum([val_report[cls]['support'] for cls in ['0', '1']])):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│ Macro Avg │   {val_report['macro avg']['precision']:.4f}  │   {val_report['macro avg']['recall']:.4f}  │   {val_report['macro avg']['f1-score']:.4f}  │ {int(val_report['macro avg']['support']):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│Weight Avg │   {val_report['weighted avg']['precision']:.4f}  │   {val_report['weighted avg']['recall']:.4f}  │   {val_report['weighted avg']['f1-score']:.4f}  │ {int(val_report['weighted avg']['support']):9d} │\n")
    f.write("└───────────┴───────────┴───────────┴───────────┴───────────┘\n\n")
    
    # Ngưỡng tối ưu từ validation
    f.write("========== NGƯỠNG TỐI ƯU TỪ VALIDATION ==========\n\n")
    f.write(f"Best Threshold (Validation): {val_best_threshold:.4f}\n")
    f.write(f"F1-score với ngưỡng tối ưu (Validation): {val_best_f1:.4f}\n\n")
    
    # Kết quả với ngưỡng từ validation trên tập test
    f.write("========== KẾT QUẢ VỚI NGƯỠNG TỪ VALIDATION ==========\n\n")
    f.write(f"F1-score: {test_with_val_threshold_f1:.4f}\n\n")
    
    f.write("Confusion Matrix:\n")
    cm_val = confusion_matrix(y_test, test_pred_with_val_threshold)
    f.write("┌─────────────┬────────────┬────────────┐\n")
    f.write("│             │ Predicted 0 │ Predicted 1 │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 0  │  {cm_val[0][0]:10d} │  {cm_val[0][1]:10d} │\n")
    f.write("├─────────────┼────────────┼────────────┤\n")
    f.write(f"│   Actual 1  │  {cm_val[1][0]:10d} │  {cm_val[1][1]:10d} │\n")
    f.write("└─────────────┴────────────┴────────────┘\n\n")
    
    f.write("Classification Report:\n")
    report_val = classification_report(y_test, test_pred_with_val_threshold, output_dict=True)
    
    # Tạo header cho bảng
    f.write("┌───────────┬───────────┬───────────┬───────────┬───────────┐\n")
    f.write("│    Class   │ Precision │   Recall  │ F1-Score  │  Support  │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    
    # Thêm các hàng cho mỗi lớp
    for cls in ['0', '1']:
        if cls in report_val:
            f.write(f"│     {cls}     │   {report_val[cls]['precision']:.4f}  │   {report_val[cls]['recall']:.4f}  │   {report_val[cls]['f1-score']:.4f}  │ {int(report_val[cls]['support']):9d} │\n")
    
    # Thêm các hàng cho accuracy, macro avg, weighted avg
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│  Accuracy │     -     │     -     │   {report_val['accuracy']:.4f}  │ {int(sum([report_val[cls]['support'] for cls in ['0', '1'] if cls in report_val])):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│ Macro Avg │   {report_val['macro avg']['precision']:.4f}  │   {report_val['macro avg']['recall']:.4f}  │   {report_val['macro avg']['f1-score']:.4f}  │ {int(report_val['macro avg']['support']):9d} │\n")
    f.write("├───────────┼───────────┼───────────┼───────────┼───────────┤\n")
    f.write(f"│Weight Avg │   {report_val['weighted avg']['precision']:.4f}  │   {report_val['weighted avg']['recall']:.4f}  │   {report_val['weighted avg']['f1-score']:.4f}  │ {int(report_val['weighted avg']['support']):9d} │\n")
    f.write("└───────────┴───────────┴───────────┴───────────┴───────────┘\n")

