import os
from pyspark.sql import SparkSession

os.environ["HADOOP_HOME"] = r"C:\hadoop"

spark = (
    SparkSession.builder.appName("MyApp")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    .master("local[2]")
    .getOrCreate()
)

# Ваш код здесь

spark.stop()
