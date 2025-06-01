import pandas as pd
# import ydata_profiling


# df = pd.read_csv('D:\\fraud_detection\\data\\data_process\\data_process.csv')
# profile = ydata_profiling.ProfileReport(df , title='report' , explorative= True)
# profile.to_file('visualize_process.html')
df = pd.read_csv('D:\\fraud_detection\\data\\raw\\data.csv')

#hiển thị Xem có bao nhiêu tài khoản nameOrig và nameDest giống nhau
print(df[df['nameOrig'] == df['nameDest']].shape[0])

