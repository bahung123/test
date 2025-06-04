import pandas as pd
import ydata_profiling 

df =pd.read_excel('data_poc/SHCLOG_POC_SVT.xlsx')
profile = ydata_profiling.ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile.to_file("data_poc/SHCLOG_POC_SVT_report.html")