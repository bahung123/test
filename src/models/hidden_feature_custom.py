from ann import ANNKNN , ANNLOF
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    ANNKNN(n_neighbors=15, method='mean'),
    ANNLOF(n_neighbors=15)
]
# Lấy decision_function output từ từng base estimator
X_train_hidden_list = []
X_val_hidden_list = []
X_test_hidden_list = []
print(f"Số lượng base estimators: {len(base_estimators)}")
for estimator in base_estimators:
    # Fit model
    estimator.fit(X_train_scaled, y_train)
    
    # Lấy decision function output
    X_train_hidden = estimator.decision_function(X_train_scaled).reshape(-1, 1)
    X_val_hidden = estimator.decision_function(X_val_scaled).reshape(-1, 1)
    X_test_hidden = estimator.decision_function(X_test_scaled).reshape(-1, 1)
    
    # Append to list
    X_train_hidden_list.append(X_train_hidden)
    X_val_hidden_list.append(X_val_hidden)
    X_test_hidden_list.append(X_test_hidden)
# Concatenate all hidden features
X_train_hidden = np.concatenate(X_train_hidden_list, axis=1)
X_val_hidden = np.concatenate(X_val_hidden_list, axis=1)
X_test_hidden = np.concatenate(X_test_hidden_list, axis=1)
# Convert to DataFrame
X_train_hidden_df = pd.DataFrame(X_train_hidden, columns=[f'feature_{i}' for i in range(X_train_hidden.shape[1])])
X_val_hidden_df = pd.DataFrame(X_val_hidden, columns=[f'feature_{i}' for i in range(X_val_hidden.shape[1])])
X_test_hidden_df = pd.DataFrame(X_test_hidden, columns=[f'feature_{i}' for i in range(X_test_hidden.shape[1])])
# Lưu các hidden features vào file CSV
X_train_hidden_df.to_csv('/home/hung/fraud_detection/data/test/X_train_hidden.csv', index=False)
X_val_hidden_df.to_csv('/home/hung/fraud_detection/data/test/X_val_hidden.csv', index=False)
X_test_hidden_df.to_csv('/home/hung/fraud_detection/data/test/X_test_hidden.csv', index=False)
print("Hidden features đã được lưu vào file CSV thành công.")
