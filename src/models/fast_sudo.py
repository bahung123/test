import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
import time

# Load data
train_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/train.csv')
test_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/test.csv')
val_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/val.csv')

# Lấy tất cả mẫu fraud
fraud_df = train_df[train_df['isFraud'] == 1]

# Lấy mẫu non-fraud 
non_fraud_df = train_df[train_df['isFraud'] == 0].sample(frac=0.5, random_state=42)

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

# Khởi tạo các base estimators anomaly detection
base_estimators = [
    IForest(n_estimators=20),
    HBOS(),
    OCSVM(nu=0.1),
    KNN(n_neighbors=15),
    LOF(n_neighbors=15)
]

# Lấy decision_function output từ từng base estimator
X_train_hidden_list = []
X_val_hidden_list = []
X_test_hidden_list = []

print(f"Số lượng base estimators: {len(base_estimators)}")

start_time = time.time()

for i, estimator in enumerate(base_estimators):
    print(f"Fit estimator {i}: {type(estimator).__name__}")
    estimator.fit(X_train_scaled)
    
    # Dùng decision_function thống nhất cho train/val/test
    train_scores = estimator.decision_function(X_train_scaled)
    val_scores = estimator.decision_function(X_val_scaled)
    test_scores = estimator.decision_function(X_test_scaled)
    
    X_train_hidden_list.append(train_scores)
    X_val_hidden_list.append(val_scores)
    X_test_hidden_list.append(test_scores)
    
    print(f"  - Shape decision scores train: {train_scores.shape}")

# Gộp các hidden features từ các base estimators
X_train_hidden = np.column_stack(X_train_hidden_list)
X_val_hidden = np.column_stack(X_val_hidden_list)
X_test_hidden = np.column_stack(X_test_hidden_list)

# Scale hidden features riêng
hidden_scaler = StandardScaler()
X_train_hidden_scaled = hidden_scaler.fit_transform(X_train_hidden)
X_val_hidden_scaled = hidden_scaler.transform(X_val_hidden)
X_test_hidden_scaled = hidden_scaler.transform(X_test_hidden)

# Kết hợp features gốc và features ẩn
X_train_combined = np.hstack([X_train_scaled, X_train_hidden_scaled])
X_val_combined = np.hstack([X_val_scaled, X_val_hidden_scaled])
X_test_combined = np.hstack([X_test_scaled, X_test_hidden_scaled])

end_time = time.time()
print(f"Thời gian fit và tạo hidden features: {end_time - start_time:.2f} giây")

# Tạo tên cột cho dataframe kết hợp
n_features = X_train_scaled.shape[1]
n_hidden = X_train_hidden_scaled.shape[1]

original_feature_cols = [f"f_{i}" for i in range(n_features)]
hidden_feature_cols = [f"hidden_{i}" for i in range(n_hidden)]
all_cols = original_feature_cols + hidden_feature_cols

# Tạo dataframe kết hợp và thêm label isFraud
train_combined_df = pd.DataFrame(X_train_combined, columns=all_cols)
train_combined_df['isFraud'] = y_train.values

val_combined_df = pd.DataFrame(X_val_combined, columns=all_cols)
val_combined_df['isFraud'] = y_val.values

test_combined_df = pd.DataFrame(X_test_combined, columns=all_cols)
test_combined_df['isFraud'] = y_test.values

# Lưu ra file CSV
train_combined_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/train_combined_half.csv', index=False)
val_combined_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/val_combined_half.csv', index=False)
test_combined_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/test_combined_half.csv', index=False)

print("Dữ liệu đã được lưu vào các file CSV.")

# In shape dữ liệu cuối cùng
print(f"Số lượng feature ẩn cuối cùng: {n_hidden}")
print(f"Shape X_train_hidden: {X_train_hidden.shape}")
print(f"Shape X_val_hidden: {X_val_hidden.shape}")
print(f"Shape X_test_hidden: {X_test_hidden.shape}")
print(f"Shape X_train_combined: {X_train_combined.shape}")
