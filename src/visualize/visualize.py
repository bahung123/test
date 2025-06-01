import pandas as pd
import ydata_profiling
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv('/home/hung/fraud_detection/train.csv')
profile = ydata_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("/home/hung/fraud_detection/train_report.html")







