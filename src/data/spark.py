from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType

# Khởi tạo SparkSession
spark = SparkSession.builder.appName("FraudDetectionPreprocessing").getOrCreate()

# Đọc dữ liệu
df = spark.read.csv('/home/hung/fraud_detection/data/raw/data.csv', header=True, inferSchema=True)

# Giữ lại chỉ các loại giao dịch có gian lận (CASH_OUT, TRANSFER)
df = df.filter(col('type').isin(['CASH_OUT','TRANSFER']))

# Gán nhãn (0: CASH_OUT, 1: TRANSFER)
df = df.withColumn('type', when(col('type') == 'CASH_OUT', 0).otherwise(1))

# Bỏ đi những giá trị có amount = 0
df = df.filter(col('amount') > 0)

# Tính sự khác biệt số dư
df = df.withColumn('balance_diff_Org', col('oldbalanceOrg') - col('newbalanceOrig'))
df = df.withColumn('balance_diff_Dest', col('oldbalanceDest') - col('newbalanceDest'))

# Xử lý nameOrig và nameDest: loại bỏ 'C' rồi chuyển sang số nguyên
from pyspark.sql.functions import regexp_replace

df = df.withColumn('nameOrig', regexp_replace(col('nameOrig'), 'C', '').cast(IntegerType()))
df = df.withColumn('nameDest', regexp_replace(col('nameDest'), 'C', '').cast(IntegerType()))

# Chuẩn hóa nameOrig và nameDest bằng StandardScaler (cần VectorAssembler để gom thành vector)
assembler = VectorAssembler(inputCols=['nameOrig'], outputCol='nameOrig_vec')
scaler = StandardScaler(inputCol='nameOrig_vec', outputCol='nameOrig_scaled')

assembler2 = VectorAssembler(inputCols=['nameDest'], outputCol='nameDest_vec')
scaler2 = StandardScaler(inputCol='nameDest_vec', outputCol='nameDest_scaled')

pipeline = Pipeline(stages=[assembler, scaler, assembler2, scaler2])
model = pipeline.fit(df)
df = model.transform(df)

# Thay thế cột nameOrig, nameDest bằng cột đã chuẩn hóa (lấy phần tử đầu vector)
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

get_first_element = udf(lambda v: float(v[0]), DoubleType())

df = df.withColumn('nameOrig', get_first_element(col('nameOrig_scaled')))
df = df.withColumn('nameDest', get_first_element(col('nameDest_scaled')))

# Bỏ các cột trung gian vector
df = df.drop('nameOrig_vec', 'nameOrig_scaled', 'nameDest_vec', 'nameDest_scaled')

# lưu dữ liệu đã xử lý chuyển sang định dạng Parquet
df.write.parquet('/home/hung/fraud_detection/data/processed/data.parquet', mode='overwrite')

# Kết thúc SparkSession
spark.stop()
