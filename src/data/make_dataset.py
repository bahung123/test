import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# Đọc dữ liệu
df = pd.read_csv('/home/hung/fraud_detection/data/raw/data.csv')

# Giữ lại chỉ các loại giao dịch có gian lận (CASH_OUT, TRANSFER)
df = df[df['type'].isin(['CASH_OUT','TRANSFER'])]

# Gán nhãn (0: CASH_OUT, 1: TRANSFER)
df['type'] = df['type'].map({'CASH_OUT': 0, 'TRANSFER': 1})


#bỏ đi những giá trị có amount=0
df = df[df['amount']>0].reset_index(drop=True)


# Tính sự khác biệt số dư
df['balance_diff_Org'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balance_diff_Dest'] = df['oldbalanceDest'] - df['newbalanceDest']

namescaler = StandardScaler()

# Chuẩn hoá nameOrg và nameDest
df['nameOrig'] = df['nameOrig'].str.replace('C', '').astype(int)
df['nameDest'] = df['nameDest'].str.replace('C', '').astype(int)
df['nameOrig'] = namescaler.fit_transform(df[['nameOrig']])
df['nameDest'] = namescaler.fit_transform(df[['nameDest']])

# chia dữ liệu thành train và test và validation
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'])
train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train['type'])

# lưu dữ liệu vào file csv
train.to_csv('/home/hung/fraud_detection/data/data_process/train.csv', index=False)
test.to_csv('/home/hung/fraud_detection/data/data_process/test.csv', index=False)
val.to_csv('/home/hung/fraud_detection/data/data_process/val.csv', index=False)




