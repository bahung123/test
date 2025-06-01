import pandas as pd
import os

train_df = pd.read_parquet('/home/hung/fraud_detection/train_epoch')
test_df = pd.read_parquet('/home/hung/fraud_detection/test_data')
val_df = pd.read_parquet('/home/hung/fraud_detection/val_data')

# Bỏ các cột dir0
train_df = train_df.drop(columns=['dir0'])

# Tạo thư mục nếu chưa tồn tại
os.makedirs('/home/hung/fraud_detection/data/data_processed_csv', exist_ok=True)

# Lưu lại thành các file csv
train_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/train.csv', index=False)
test_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/test.csv', index=False)
val_df.to_csv('/home/hung/fraud_detection/data/data_processed_csv/val.csv', index=False)
