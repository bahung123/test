import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.lscp import LSCP as PyODLSCP
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.cblof import CBLOF
from pyod.utils.data import evaluate_print
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, auc,
                           average_precision_score)

print("LSCP (Locally Selective Combination in Parallel Outlier Ensembles) Model")
print("=" * 60)

# Đọc dữ liệu
print("Loading data...")
train_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/train.csv")
test_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/test.csv")
val_df = pd.read_csv("/home/hung/fraud_detection/data/data_processed_csv/val.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Validation shape: {val_df.shape}")

# Split features and target
X_train = train_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_train = train_df['isFraud']
X_test = test_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_test = test_df['isFraud']
X_val = val_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_val = val_df['isFraud']

print(f"Feature columns: {X_train.columns.tolist()}")
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test fraud rate: {y_test.mean():.4f}")
print(f"Val fraud rate: {y_val.mean():.4f}")

# Chuẩn hóa dữ liệu
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các base detectors cho LSCP
print("\nInitializing base detectors...")
detector_list = [
    LOF(n_neighbors=20, contamination=0.1),
    LOF(n_neighbors=35, contamination=0.1), 
    IForest(n_estimators=100, contamination=0.1, random_state=42),
    IForest(n_estimators=200, contamination=0.1, random_state=123),
    KNN(n_neighbors=10, contamination=0.1),
    KNN(n_neighbors=20, contamination=0.1),
    OCSVM(contamination=0.1),
    CBLOF(contamination=0.1, random_state=42)
]

print(f"Number of base detectors: {len(detector_list)}")

# Khởi tạo LSCP model
print("\nInitializing LSCP model...")
time_start = time.time()

lscp_model = PyODLSCP(
    detector_list=detector_list,
    contamination=0.1,  # tỷ lệ outlier dự kiến
    random_state=42
)

# Huấn luyện model
print("Training LSCP model...")
lscp_model.fit(X_train_scaled)
training_time = time.time() - time_start
print(f"Training completed in {training_time:.2f} seconds")

# Dự đoán trên tập validation
print("\nMaking predictions...")
pred_time_start = time.time()

# Outlier scores và labels
y_val_scores = lscp_model.decision_function(X_val_scaled)
y_val_pred = lscp_model.predict(X_val_scaled)

y_test_scores = lscp_model.decision_function(X_test_scaled)
y_test_pred = lscp_model.predict(X_test_scaled)

prediction_time = time.time() - pred_time_start
print(f"Prediction completed in {prediction_time:.2f} seconds")

# Đánh giá model trên validation set
print("\n" + "="*50)
print("VALIDATION SET RESULTS")
print("="*50)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_roc_auc = roc_auc_score(y_val, y_val_scores)

print(f"Accuracy: {val_accuracy:.4f}")
print(f"ROC AUC: {val_roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix:")
val_cm = confusion_matrix(y_val, y_val_pred)
print(val_cm)

# Đánh giá model trên test set
print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_scores)

print(f"Accuracy: {test_accuracy:.4f}")
print(f"ROC AUC: {test_roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix:")
test_cm = confusion_matrix(y_test, y_test_pred)
print(test_cm)

# Precision-Recall curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_scores)
pr_auc_val = auc(recall_val, precision_val)

precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_scores)
pr_auc_test = auc(recall_test, precision_test)

print(f"\nPR AUC (Validation): {pr_auc_val:.4f}")
print(f"PR AUC (Test): {pr_auc_test:.4f}")

# Tính Average Precision Score
ap_val = average_precision_score(y_val, y_val_scores)
ap_test = average_precision_score(y_test, y_test_scores)

print(f"Average Precision (Validation): {ap_val:.4f}")
print(f"Average Precision (Test): {ap_test:.4f}")

# Lưu kết quả vào file
output_dir = "/home/hung/fraud_detection/src/output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

results_text = f"""LSCP Model Results
==================

Model Configuration:
- Base Detectors: {len(detector_list)}
- Contamination Rate: 0.1
- Training Time: {training_time:.2f} seconds
- Prediction Time: {prediction_time:.2f} seconds

Validation Set Results:
- Accuracy: {val_accuracy:.4f}
- ROC AUC: {val_roc_auc:.4f}
- PR AUC: {pr_auc_val:.4f}
- Average Precision: {ap_val:.4f}

Test Set Results:
- Accuracy: {test_accuracy:.4f}
- ROC AUC: {test_roc_auc:.4f}
- PR AUC: {pr_auc_test:.4f}
- Average Precision: {ap_test:.4f}

Base Detectors Used:
{[type(detector).__name__ for detector in detector_list]}

Classification Report (Test):
{classification_report(y_test, y_test_pred)}

Confusion Matrix (Test):
{test_cm}
"""

with open(f"{output_dir}/lscp_metrics.txt", "w") as f:
    f.write(results_text)

print(f"\nResults saved to {output_dir}/lscp_metrics.txt")

# Visualization
print("\nCreating visualizations...")

# Tạo thư mục images cho LSCP nếu chưa có
images_dir = "/home/hung/fraud_detection/src/images/lscp"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# 1. ROC Curves comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_scores)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores)

plt.plot(fpr_val, tpr_val, label=f'Validation (AUC = {val_roc_auc:.3f})', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LSCP ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Precision-Recall Curves
plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val, label=f'Validation (AUC = {pr_auc_val:.3f})', linewidth=2)
plt.plot(recall_test, precision_test, label=f'Test (AUC = {pr_auc_test:.3f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('LSCP Precision-Recall Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{images_dir}/lscp_roc_pr_curves.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Score distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y_val_scores[y_val == 0], bins=50, alpha=0.7, label='Normal', density=True)
plt.hist(y_val_scores[y_val == 1], bins=50, alpha=0.7, label='Fraud', density=True)
plt.xlabel('Outlier Score')
plt.ylabel('Density')
plt.title('LSCP Score Distribution (Validation)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_test_scores[y_test == 0], bins=50, alpha=0.7, label='Normal', density=True)
plt.hist(y_test_scores[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
plt.xlabel('Outlier Score')
plt.ylabel('Density')
plt.title('LSCP Score Distribution (Test)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{images_dir}/lscp_score_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Confusion matrices heatmap
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix - Validation')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix - Test')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(f"{images_dir}/lscp_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("LSCP MODEL TRAINING AND EVALUATION COMPLETED!")
print("="*60)
print(f"Model training time: {training_time:.2f} seconds")
print(f"Best Test ROC AUC: {test_roc_auc:.4f}")
print(f"Best Test PR AUC: {pr_auc_test:.4f}")
print(f"Results saved to: {output_dir}/lscp_metrics.txt")
print(f"Visualizations saved to: {images_dir}/")
print("="*60)

