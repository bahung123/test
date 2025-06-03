import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
import time
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score, average_precision_score

# Load data
train_df = pd.read_csv('data/data_processed_csv/train.csv')
test_df = pd.read_csv('data/data_processed_csv/test.csv')
val_df = pd.read_csv('data/data_processed_csv/val.csv')

# Lấy tất cả mẫu fraud
fraud_df = train_df[train_df['isFraud'] == 1]

# Lấy mẫu non-fraud 
non_fraud_df = train_df[train_df['isFraud'] == 0].sample(frac=0.1, random_state=42)

# Gộp lại tạo tập train cân bằng hơn
train_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)

# Shuffle lại dữ liệu train
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Tách features và target
X_train = train_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_train = train_df['isFraud']

X_val = val_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_val = val_df['isFraud']

X_test = test_df.drop(['isFraud','isFlaggedFraud'], axis=1)
y_test = test_df['isFraud']

# Scale dữ liệu gốc
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("=== BƯỚC 1: TẠO HIDDEN FEATURES VỚI ANOMALY DETECTION ===")

# Khởi tạo các base estimators anomaly detection với diversity cao hơn
anomaly_estimators = [
    # Isolation Forest với các cấu hình khác nhau
    IForest(n_estimators=100, contamination=0.03, random_state=42, max_features=0.8),
    IForest(n_estimators=200, contamination=0.05, random_state=43, max_features=0.6),
    IForest(n_estimators=150, contamination=0.08, random_state=44, max_features=1.0),
    
    # HBOS - Histogram-based Outlier Detection với cấu hình đa dạng
    HBOS(contamination=0.03, n_bins=15, alpha=0.1),
    HBOS(contamination=0.05, n_bins=25, alpha=0.1),
    HBOS(contamination=0.08, n_bins=20, alpha=0.1),
    ]

# Lấy decision_function output từ từng base estimator
X_train_hidden_list = []
X_val_hidden_list = []
X_test_hidden_list = []

print(f"Số lượng anomaly detectors: {len(anomaly_estimators)}")

start_time = time.time()

for i, estimator in enumerate(anomaly_estimators):
    try:
        print(f"Fit anomaly detector {i}: {type(estimator).__name__}")
        estimator.fit(X_train_scaled)
        
        # Dùng decision_function thống nhất cho train/val/test
        train_scores = estimator.decision_function(X_train_scaled)
        val_scores = estimator.decision_function(X_val_scaled)
        test_scores = estimator.decision_function(X_test_scaled)
        
        X_train_hidden_list.append(train_scores)
        X_val_hidden_list.append(val_scores)
        X_test_hidden_list.append(test_scores)
        
        print(f"  - Shape decision scores train: {train_scores.shape}")
        
    except Exception as e:
        print(f"  - ERROR với {type(estimator).__name__}: {e}")
        # Tạo random scores làm fallback
        train_scores = np.random.normal(0, 1, X_train_scaled.shape[0])
        val_scores = np.random.normal(0, 1, X_val_scaled.shape[0])
        test_scores = np.random.normal(0, 1, X_test_scaled.shape[0])
        
        X_train_hidden_list.append(train_scores)
        X_val_hidden_list.append(val_scores)
        X_test_hidden_list.append(test_scores)

# Gộp các hidden features từ các base estimators
X_train_hidden = np.column_stack(X_train_hidden_list)
X_val_hidden = np.column_stack(X_val_hidden_list)
X_test_hidden = np.column_stack(X_test_hidden_list)

# Sử dụng MinMaxScaler để chuẩn hóa giá trị về khoảng [0, 1]
hidden_scaler = MinMaxScaler()
X_train_hidden_scaled = hidden_scaler.fit_transform(X_train_hidden)
X_val_hidden_scaled = hidden_scaler.transform(X_val_hidden)
X_test_hidden_scaled = hidden_scaler.transform(X_test_hidden)

# Kết hợp features gốc và features ẩn
X_train_combined = np.hstack([X_train_scaled, X_train_hidden_scaled])
X_val_combined = np.hstack([X_val_scaled, X_val_hidden_scaled])
X_test_combined = np.hstack([X_test_scaled, X_test_hidden_scaled])

# Lưu combined features
train_combined_df = pd.DataFrame(X_train_combined, columns=[f"f_{i}" for i in range(X_train_combined.shape[1])])
val_combined_df = pd.DataFrame(X_val_combined, columns=[f"f_{i}" for i in range(X_val_combined.shape[1])])
test_combined_df = pd.DataFrame(X_test_combined, columns=[f"f_{i}" for i in range(X_test_combined.shape[1])])

train_combined_df['isFraud'] = y_train.values
val_combined_df['isFraud'] = y_val.values
test_combined_df['isFraud'] = y_test.values

train_combined_df.to_csv('data/train_combined.csv', index=False)
val_combined_df.to_csv('data/val_combined.csv', index=False)
test_combined_df.to_csv('data/test_combined.csv', index=False)

end_time = time.time()
print(f"Thời gian tạo hidden features: {end_time - start_time:.2f} giây")

print("\n=== BƯỚC 2: SOFT VOTING VỚI CÁC MÔ HÌNH KHÔNG GIÁM SÁT ===")

def convert_scores_to_probabilities(train_scores, val_scores, test_scores, method='sigmoid'):
    """
    Chuyển đổi decision scores thành xác suất với scaling thống nhất
    Fit scaler trên train và apply cho val/test để tránh data leakage
    """
    if method == 'sigmoid':
        # Fit scaler trên train scores để đảm bảo consistency
        scaler = MinMaxScaler(feature_range=(-5, 5))
        train_scores_scaled = scaler.fit_transform(train_scores.reshape(-1, 1)).flatten()
        val_scores_scaled = scaler.transform(val_scores.reshape(-1, 1)).flatten()
        test_scores_scaled = scaler.transform(test_scores.reshape(-1, 1)).flatten()
        
        # Apply sigmoid
        train_probs = 1 / (1 + np.exp(-train_scores_scaled))
        val_probs = 1 / (1 + np.exp(-val_scores_scaled))
        test_probs = 1 / (1 + np.exp(-test_scores_scaled))
        
        return train_probs, val_probs, test_probs
    elif method == 'minmax':
        # MinMax normalization về [0, 1]
        scaler = MinMaxScaler()
        train_probs = scaler.fit_transform(train_scores.reshape(-1, 1)).flatten()
        val_probs = scaler.transform(val_scores.reshape(-1, 1)).flatten()
        test_probs = scaler.transform(test_scores.reshape(-1, 1)).flatten()
        
        return train_probs, val_probs, test_probs
    else:
        # Default: sigmoid
        scaler = MinMaxScaler(feature_range=(-5, 5))
        train_scores_scaled = scaler.fit_transform(train_scores.reshape(-1, 1)).flatten()
        val_scores_scaled = scaler.transform(val_scores.reshape(-1, 1)).flatten()
        test_scores_scaled = scaler.transform(test_scores.reshape(-1, 1)).flatten()
        
        train_probs = 1 / (1 + np.exp(-train_scores_scaled))
        val_probs = 1 / (1 + np.exp(-val_scores_scaled))
        test_probs = 1 / (1 + np.exp(-test_scores_scaled))
        
        return train_probs, val_probs, test_probs

# Sử dụng lại các anomaly estimators đã fit từ BƯỚC 1
print(f"Số lượng unsupervised models: {len(anomaly_estimators)}")

X_train_probs_list = []
X_val_probs_list = []
X_test_probs_list = []

start_time = time.time()

# Lấy decision scores và chuyển thành xác suất từ từng mô hình không giám sát
for i, estimator in enumerate(anomaly_estimators):
    print(f"Convert scores to probabilities từ {type(estimator).__name__}")
    
    # Lấy decision scores (đã tính ở BƯỚC 1)
    train_scores = X_train_hidden_list[i]
    val_scores = X_val_hidden_list[i] 
    test_scores = X_test_hidden_list[i]
    
    # Chuyển scores thành xác suất bằng sigmoid với scaling thống nhất
    train_probs, val_probs, test_probs = convert_scores_to_probabilities(
        train_scores, val_scores, test_scores, method='sigmoid'
    )
    
    X_train_probs_list.append(train_probs)
    X_val_probs_list.append(val_probs)
    X_test_probs_list.append(test_probs)
    
    print(f"  - Scores range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")
    print(f"  - Probs range: [{train_probs.min():.3f}, {train_probs.max():.3f}]")

# Gộp các xác suất từ các unsupervised models
X_train_probs = np.column_stack(X_train_probs_list)
X_val_probs = np.column_stack(X_val_probs_list)
X_test_probs = np.column_stack(X_test_probs_list)

# Tính trung bình xác suất (soft voting từ unsupervised models)
train_preds = np.mean(X_train_probs, axis=1)
val_preds = np.mean(X_val_probs, axis=1)
test_preds = np.mean(X_test_probs, axis=1)

print(f"Soft voting probabilities range:")
print(f"  - Train: [{train_preds.min():.3f}, {train_preds.max():.3f}]")
print(f"  - Val: [{val_preds.min():.3f}, {val_preds.max():.3f}]")
print(f"  - Test: [{test_preds.min():.3f}, {test_preds.max():.3f}]")

print("\n=== BƯỚC 3: TỐI ƯU THRESHOLD TRÊN VALIDATION SET ===")

# Tìm threshold tốt nhất trên validation set bằng Grid Search chi tiết hơn
thresholds = np.arange(0.05, 0.95, 0.005)  # Tăng độ chi tiết và mở rộng range
best_threshold = 0.5
best_val_score = 0

# Sử dụng metric tổng hợp được cải thiện: PR-AUC + F1-Score
from sklearn.metrics import average_precision_score

for threshold in thresholds:
    val_final_preds = (val_preds >= threshold).astype(int)
    
    # Tính các metrics
    val_f1 = f1_score(y_val, val_final_preds)
    val_precision = precision_score(y_val, val_final_preds, zero_division=0)
    val_recall = recall_score(y_val, val_final_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    val_pr_auc = average_precision_score(y_val, val_preds)
    
    # Metric tổng hợp: cân bằng giữa F1, Precision và PR-AUC (quan trọng cho imbalanced data)
    combined_score = (0.4 * val_f1 + 0.3 * val_precision + 0.3 * val_pr_auc)
    
    if combined_score > best_val_score:
        best_val_score = combined_score
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.3f} với validation combined score: {best_val_score:.4f}")

print("\n=== BƯỚC 4: ENSEMBLE SOFT VOTING - KẾT HỢP CÁC XÁC SUẤT ===")

# Phương pháp 1: Trung bình đơn giản (Simple Average)
ensemble_probs_avg = np.mean(X_test_probs, axis=1)

# Phương pháp 2: Trung bình có trọng số (Weighted Average) - TỐI ƯU HƠN
# Sử dụng nhiều metrics để tính trọng số

weights = []
for i in range(len(anomaly_estimators)):
    val_probs_i = X_val_probs_list[i]
    
    # Tính nhiều metrics để đánh giá model performance
    auc_i = roc_auc_score(y_val, val_probs_i)
    pr_auc_i = average_precision_score(y_val, val_probs_i)
    
    # Tìm threshold tốt nhất cho model này
    best_f1_i = 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        val_preds_thresh = (val_probs_i >= thresh).astype(int)
        f1_i = f1_score(y_val, val_preds_thresh)
        if f1_i > best_f1_i:
            best_f1_i = f1_i
    
    # Tổng hợp score để tính trọng số: cân bằng AUC, PR-AUC và F1
    model_score = (0.4 * auc_i + 0.3 * pr_auc_i + 0.3 * best_f1_i)
    weights.append(model_score)
    
weights = np.array(weights)
weights = weights / np.sum(weights)  # Normalize weights

print("Trọng số của từng mô hình (dựa trên tổng hợp AUC, PR-AUC, F1):")
for i, (estimator, weight) in enumerate(zip(anomaly_estimators, weights)):
    val_probs_i = X_val_probs_list[i]
    auc_i = roc_auc_score(y_val, val_probs_i)
    pr_auc_i = average_precision_score(y_val, val_probs_i)
    print(f"  {type(estimator).__name__:15}: {weight:.4f} (AUC:{auc_i:.3f}, PR-AUC:{pr_auc_i:.3f})")

ensemble_probs_weighted = np.average(X_test_probs, axis=1, weights=weights)

# Phương pháp 3: Hard Voting được cải thiện
hard_votes = []
for i in range(len(anomaly_estimators)):
    # Tìm threshold tốt nhất cho mô hình i trên validation set bằng F1-Score
    model_val_probs = X_val_probs_list[i]
    best_thresh_i = 0.5
    best_f1_i = 0
    
    for thresh in np.arange(0.05, 0.95, 0.02):  # Mở rộng range và tăng độ chi tiết
        val_preds_thresh = (model_val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, val_preds_thresh)
        if f1 > best_f1_i:
            best_f1_i = f1
            best_thresh_i = thresh
    
    # Apply threshold to test predictions
    hard_vote_i = (X_test_probs_list[i] >= best_thresh_i).astype(int)
    hard_votes.append(hard_vote_i)

hard_votes_matrix = np.column_stack(hard_votes)

# Sử dụng weighted hard voting thay vì simple majority
weighted_hard_votes = np.zeros(len(y_test))
for i, vote in enumerate(hard_votes):
    weighted_hard_votes += vote * weights[i]

ensemble_preds_hard = (weighted_hard_votes >= 0.5).astype(int)

print(f"\nKết quả Ensemble Probabilities:")
print(f"  - Simple Average range: [{ensemble_probs_avg.min():.3f}, {ensemble_probs_avg.max():.3f}]")
print(f"  - Weighted Average range: [{ensemble_probs_weighted.min():.3f}, {ensemble_probs_weighted.max():.3f}]")
print(f"  - Hard Voting predictions: {np.sum(ensemble_preds_hard)} fraud cases out of {len(ensemble_preds_hard)}")

print("\n=== BƯỚC 5: ĐÁNH GIÁ TRÊN TẬP TEST ===")

# Áp dụng threshold tốt nhất cho các phương pháp ensemble
ensemble_preds_avg = (ensemble_probs_avg >= best_threshold).astype(int)
ensemble_preds_weighted = (ensemble_probs_weighted >= best_threshold).astype(int)

print("=== KẾT QUẢ CHI TIẾT CỦA CÁC PHƯƠNG PHÁP ENSEMBLE ===")

# 1. Simple Average Ensemble
print("\n1. SIMPLE AVERAGE ENSEMBLE:")
print(f"   Accuracy: {accuracy_score(y_test, ensemble_preds_avg):.4f}")
print(f"   AUC: {roc_auc_score(y_test, ensemble_probs_avg):.4f}")
print(f"   F1-Score: {f1_score(y_test, ensemble_preds_avg):.4f}")
print(f"   Precision: {precision_score(y_test, ensemble_preds_avg):.4f}")
print(f"   Recall: {recall_score(y_test, ensemble_preds_avg):.4f}")

cm_avg = confusion_matrix(y_test, ensemble_preds_avg)
print("   Confusion Matrix:")
print(f"   [[TN={cm_avg[0,0]}, FP={cm_avg[0,1]}],")
print(f"    [FN={cm_avg[1,0]}, TP={cm_avg[1,1]}]]")

# 2. Weighted Average Ensemble  
print("\n2. WEIGHTED AVERAGE ENSEMBLE:")
print(f"   Accuracy: {accuracy_score(y_test, ensemble_preds_weighted):.4f}")
print(f"   AUC: {roc_auc_score(y_test, ensemble_probs_weighted):.4f}")
print(f"   F1-Score: {f1_score(y_test, ensemble_preds_weighted):.4f}")
print(f"   Precision: {precision_score(y_test, ensemble_preds_weighted):.4f}")
print(f"   Recall: {recall_score(y_test, ensemble_preds_weighted):.4f}")

cm_weighted = confusion_matrix(y_test, ensemble_preds_weighted)
print("   Confusion Matrix:")
print(f"   [[TN={cm_weighted[0,0]}, FP={cm_weighted[0,1]}],")
print(f"    [FN={cm_weighted[1,0]}, TP={cm_weighted[1,1]}]]")

# 3. Hard Voting Ensemble
print("\n3. HARD VOTING ENSEMBLE:")
print(f"   Accuracy: {accuracy_score(y_test, ensemble_preds_hard):.4f}")
print(f"   F1-Score: {f1_score(y_test, ensemble_preds_hard):.4f}")
print(f"   Precision: {precision_score(y_test, ensemble_preds_hard):.4f}")
print(f"   Recall: {recall_score(y_test, ensemble_preds_hard):.4f}")

cm_hard = confusion_matrix(y_test, ensemble_preds_hard)
print("   Confusion Matrix:")
print(f"   [[TN={cm_hard[0,0]}, FP={cm_hard[0,1]}],")
print(f"    [FN={cm_hard[1,0]}, TP={cm_hard[1,1]}]]")

# Áp dụng threshold tốt nhất cho các phương pháp ensemble
train_final_preds = (train_preds >= best_threshold).astype(int)
val_final_preds = (val_preds >= best_threshold).astype(int)
test_final_preds = ensemble_preds_weighted  # Sử dụng weighted average làm kết quả chính

print("\n=== KẾT QUẢ CUỐI CÙNG - SOFT VOTING ENSEMBLE ===")

# Tính các metrics cho soft voting - SỬA LỖI: sử dụng ensemble_probs_weighted thay vì test_preds
soft_voting_acc = accuracy_score(y_test, test_final_preds)
soft_voting_auc = roc_auc_score(y_test, ensemble_probs_weighted)  # SỬA: dùng ensemble probabilities
soft_voting_f1 = f1_score(y_test, test_final_preds)
soft_voting_precision = precision_score(y_test, test_final_preds)
soft_voting_recall = recall_score(y_test, test_final_preds)

print(f"SOFT VOTING KẾT QUẢ:")
print(f"  - Accuracy: {soft_voting_acc:.4f}")
print(f"  - AUC: {soft_voting_auc:.4f}")
print(f"  - PR-AUC: {average_precision_score(y_test, ensemble_probs_weighted):.4f}")
print(f"  - F1-Score: {soft_voting_f1:.4f}")
print(f"  - Precision: {soft_voting_precision:.4f}")
print(f"  - Recall: {soft_voting_recall:.4f}")

# Confusion Matrix
cm_soft_voting = confusion_matrix(y_test, test_final_preds)
print(f"\nConfusion Matrix:")
print(f"  [[TN={cm_soft_voting[0,0]}, FP={cm_soft_voting[0,1]}],")
print(f"   [FN={cm_soft_voting[1,0]}, TP={cm_soft_voting[1,1]}]]")

print(f"\nCác thông số chi tiết:")
print(f"  - True Negatives (TN): {cm_soft_voting[0,0]}")
print(f"  - False Positives (FP): {cm_soft_voting[0,1]}")
print(f"  - False Negatives (FN): {cm_soft_voting[1,0]}")
print(f"  - True Positives (TP): {cm_soft_voting[1,1]}")

print(f"\nTỷ lệ dự đoán:")
print(f"  - Tổng số mẫu test: {len(y_test)}")
print(f"  - Số mẫu fraud thực tế: {np.sum(y_test)}")
print(f"  - Số mẫu fraud dự đoán: {np.sum(test_final_preds)}")
print(f"  - Tỷ lệ fraud thực tế: {np.sum(y_test)/len(y_test)*100:.2f}%")
print(f"  - Tỷ lệ fraud dự đoán: {np.sum(test_final_preds)/len(test_final_preds)*100:.2f}%")

print("\nClassification Report (Test Set - Soft Voting):")
print(classification_report(y_test, test_final_preds))

# Lưu kết quả soft voting
print(f"\nLưu kết quả vào file CSV...")

# Lưu kết quả cuối cùng với ensemble probabilities
train_final_df = pd.DataFrame(X_train_combined, columns=[f"f_{i}" for i in range(X_train_combined.shape[1])])
train_final_df['isFraud'] = y_train.values
train_final_df['soft_voting_probs'] = np.average(X_train_probs, axis=1, weights=weights)  # SỬA: dùng weighted ensemble
train_final_df['soft_voting_preds'] = train_final_preds

val_final_df = pd.DataFrame(X_val_combined, columns=[f"f_{i}" for i in range(X_val_combined.shape[1])])
val_final_df['isFraud'] = y_val.values
val_final_df['soft_voting_probs'] = np.average(X_val_probs, axis=1, weights=weights)  # SỬA: dùng weighted ensemble
val_final_df['soft_voting_preds'] = val_final_preds

test_final_df = pd.DataFrame(X_test_combined, columns=[f"f_{i}" for i in range(X_test_combined.shape[1])])
test_final_df['isFraud'] = y_test.values
test_final_df['soft_voting_probs'] = ensemble_probs_weighted  # SỬA: dùng ensemble probabilities
test_final_df['soft_voting_preds'] = test_final_preds

train_final_df.to_csv('data/data_processed_csv/train_soft_voting_final.csv', index=False)
val_final_df.to_csv('data/data_processed_csv/val_soft_voting_final.csv', index=False)
test_final_df.to_csv('data/data_processed_csv/test_soft_voting_final.csv', index=False)

print(f"Đã lưu kết quả vào:")
print(f"  - data/data_processed_csv/train_soft_voting_final.csv")
print(f"  - data/data_processed_csv/val_soft_voting_final.csv") 
print(f"  - data/data_processed_csv/test_soft_voting_final.csv")

end_time = time.time()
print(f"\nThời gian hoàn thành: {end_time - start_time:.2f} giây")
