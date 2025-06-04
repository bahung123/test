from ann import ANNKNN , ANNLOF
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
import time

# Load data
train_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/train.csv')
test_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/test.csv')
val_df = pd.read_csv('/home/hung/fraud_detection/data/data_processed_csv/val.csv')

# Tách features và target
X_train = train_df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_train = train_df['isFraud']
X_val = val_df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_val = val_df['isFraud']
X_test = test_df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y_test = test_df['isFraud']

# Scale dữ liệu gốc
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# Khởi tạo các base estimators anomaly detection
base_estimators = [

    IForest(n_estimators=10),
    IForest(n_estimators=20),
    IForest(n_estimators=50),
    IForest(n_estimators=70),
    IForest(n_estimators=100),
    IForest(n_estimators=150),
    IForest(n_estimators=200),

    HBOS(5),
    HBOS(10),
    HBOS(15),
    HBOS(20),
    HBOS(25),
    HBOS(30),
    HBOS(50),

    ANNKNN(n_neighbors=1),
    ANNKNN(n_neighbors=3),
    ANNKNN(n_neighbors=5),
    ANNKNN(n_neighbors=10),
    ANNKNN(n_neighbors=20),
    ANNKNN(n_neighbors=30),
    ANNKNN(n_neighbors=40),
    ANNKNN(n_neighbors=50),

    ANNLOF(n_neighbors=1),
    ANNLOF(n_neighbors=3),
    ANNLOF(n_neighbors=5),
    ANNLOF(n_neighbors=10),
    ANNLOF(n_neighbors=20),
    ANNLOF(n_neighbors=30),
    ANNLOF(n_neighbors=40),
    ANNLOF(n_neighbors=50)
]

# Lấy decision_function output từ từng base estimator
X_train_hidden_list = []
X_val_hidden_list = []
X_test_hidden_list = []
print(f"Số lượng base estimators: {len(base_estimators)}")

start_time = time.time()
for i, estimator in enumerate(base_estimators):
    print(f"Training base estimator {i + 1}/{len(base_estimators)}: {estimator.__class__.__name__}")
    
    # Fit the model
    estimator.fit(X_train_scaled)
    
    # Get the decision function output
    X_train_hidden = estimator.decision_function(X_train_scaled).reshape(-1, 1)
    X_val_hidden = estimator.decision_function(X_val_scaled).reshape(-1, 1)
    X_test_hidden = estimator.decision_function(X_test_scaled).reshape(-1, 1)
    
    # Append to the list
    X_train_hidden_list.append(X_train_hidden)
    X_val_hidden_list.append(X_val_hidden)
    X_test_hidden_list.append(X_test_hidden)
    print(f"  - Shape decision scores train: {X_train_hidden.shape}")

# Gộp các hidden features từ các base estimators
X_train_hidden = np.hstack(X_train_hidden_list)
X_val_hidden = np.hstack(X_val_hidden_list)
X_test_hidden = np.hstack(X_test_hidden_list)
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
feature_names = [f"feature_{i}" for i in range(n_features)] + [f"hidden_feature_{i}" for i in range(n_hidden)]
# Tạo DataFrame kết hợp
X_train_combined_df = pd.DataFrame(X_train_combined, columns=feature_names)
X_val_combined_df = pd.DataFrame(X_val_combined, columns=feature_names)
X_test_combined_df = pd.DataFrame(X_test_combined, columns=feature_names)
# Thêm label isFraud
X_train_combined_df['isFraud'] = y_train.values
X_val_combined_df['isFraud'] = y_val.values
X_test_combined_df['isFraud'] = y_test.values
# Lưu DataFrame kết hợp
X_train_combined_df.to_csv('/home/hung/fraud_detection/data/test/train_combined.csv', index=False)
X_val_combined_df.to_csv('/home/hung/fraud_detection/data/test/val_combined.csv', index=False)
X_test_combined_df.to_csv('/home/hung/fraud_detection/data/test/test_combined.csv', index=False)
