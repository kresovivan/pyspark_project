"""
Простой рекомендательный движок на основе ALS
"""

from pyspark.sql.functions import regexp_extract
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when, broadcast, trim
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml.recommendation import ALS

# ============================================
# 1. СОЗДАНИЕ SPARK SESSION
# ============================================
spark = (
    SparkSession.builder.master("local[*]")
    .appName("Simple Recommendation Engine")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .getOrCreate()
)

# ============================================
# 2. ЧТЕНИЕ ДАННЫХ
# ============================================
print("Чтение данных...")

# Чтение raw данных
raw_user_artist = spark.read.text("data/user_artist_data.txt")
raw_artist_alias = spark.read.text("data/artist_alias.txt")
raw_artist_data = spark.read.text("data/artist_data.txt")

# ============================================
# 3. ПАРСИНГ user_artist
# ============================================
print("Парсинг user_artist...")
user_artist = raw_user_artist.select(
    split(col("value"), " ").getItem(0).cast(IntegerType()).alias("user"),
    split(col("value"), " ").getItem(1).cast(IntegerType()).alias("artist"),
    split(col("value"), " ").getItem(2).cast(IntegerType()).alias("count"),
)

# Фильтрация null
user_artist = (
    user_artist.filter(col("user").isNotNull())
    .filter(col("artist").isNotNull())
    .filter(col("count").isNotNull())
)

print(f"user_artist: {user_artist.count()} записей")

# ============================================
# 4. ПАРСИНГ artist_alias (с пробелами)
# ============================================
print("Парсинг artist_alias...")

artist_alias = raw_artist_alias.select(
    regexp_extract(col("value"), r"^(\d+)", 1).alias("artist_raw"),
    regexp_extract(col("value"), r"\s+(\d+)$", 1).alias("alias_raw"),
)


# Преобразуем в числа
artist_alias = (
    artist_alias.filter(col("artist_raw") != "")
    .filter(col("alias_raw") != "")
    .withColumn("artist", col("artist_raw").cast(IntegerType()))
    .withColumn("alias", col("alias_raw").cast(IntegerType()))
    .drop("artist_raw", "alias_raw")
    .filter(col("artist").isNotNull())
    .filter(col("alias").isNotNull())
)

print(f"artist_alias: {artist_alias.count()} записей")
artist_alias.show(5)


# ============================================
# 4.5. ПАРСИНГ artist_data (id и имя исполнителя)
# ============================================
print("Парсинг artist_data...")

# Шаг 1: Извлекаем строковое значение id
artist_by_id = raw_artist_data.withColumn(
    "id_str", split(col("value"), "\s+", 2).getItem(0)
)

# Шаг 2: Фильтруем строки, где id_str состоит только из цифр
artist_by_id = artist_by_id.filter(col("id_str").rlike("^[0-9]+$"))

# Шаг 3: Преобразуем в Integer
artist_by_id = artist_by_id.withColumn("id", col("id_str").cast(IntegerType()))

# Шаг 4: Добавляем колонку 'name'
artist_by_id = artist_by_id.withColumn(
    "name", split(col("value"), "\s+", 2).getItem(1).cast(StringType())
).drop("value", "id_str")

# Шаг 5: Удаляем строки с null (на всякий случай)
artist_by_id = artist_by_id.filter(col("id").isNotNull())

print(f"artist_by_id: {artist_by_id.count()} записей")
artist_by_id.show(20)


# ============================================
# 5. ПРОСТО ИСПОЛЬЗУЕМ ОРИГИНАЛЬНЫЕ ДАННЫЕ (БЕЗ ЗАМЕНЫ ALIAS)
# ============================================
print("Используем оригинальные данные без замены alias...")
train_data = user_artist

print(f"train_data: {train_data.count()} записей")
train_data.show(10)


# ============================================
# 6. ОБУЧЕНИЕ МОДЕЛИ
# ============================================
print("Обучение модели ALS...")

train_data_numeric = (
    train_data.withColumn("user", col("user").cast(IntegerType()))
    .withColumn("artist", col("artist").cast(IntegerType()))
    .withColumn("count", col("count").cast(IntegerType()))
    .na.drop()
)

print(f"train_data_numeric: {train_data_numeric.count()} записей")

"""
model = ALS(
    rank=50,           # ещё больше факторов
    maxIter=15,        # больше итераций
    regParam=0.01,     # минимальная регуляризация
    implicitPrefs=True,
    alpha=2.0,         # ↑ увеличить доверие к данным
    userCol="user",
    itemCol="artist",
    ratingCol="count",
    coldStartStrategy="drop"
).fit(train_data_numeric)
"""


model = ALS(
    rank=10,
    seed=0,
    maxIter=5,
    regParam=0.1,
    implicitPrefs=True,
    alpha=1.0,
    userCol="user",
    itemCol="artist",
    ratingCol="count",
).fit(train_data_numeric)

print("✅ Модель успешно обучена!")


# ============================================
# 8. Выборочная проверка рекомендаций
# ============================================
user_id = 2093760

existing_artist_ids = (
    train_data.filter(col("user") == user_id).select("artist").collect()
)

existing_artist_ids = [row.artist for row in existing_artist_ids]

artist_by_id.filter(col("id").isin(existing_artist_ids)).show()


# ============================================
# 9. Дадим пльзователю рекомендацию
# ============================================
user_subset = train_data.select("user").where(col("user") == user_id).distinct()

top_predictions = model.recommendForUserSubset(user_subset, 5)

top_predictions.show(10, truncate=False)


top_predictions_pandas = top_predictions.toPandas()
print(top_predictions_pandas)

recommended_artist_ids = [i[0] for i in top_predictions_pandas.recommendations[0]]
"""
Эта конструкция проходится по этому списку и из каждого элемента вытаскивает первое число (artist_id):
# Шаг 1: i = (829, 0.15556057) → берём i[0] = 829
# Шаг 2: i = (1811, 0.15442578) → берём i[0] = 1811
# Шаг 3: i = (1001819, 0.15356433) → берём i[0] = 1001819
# Шаг 4: i = (1037970, 0.14996031) → берём i[0] = 1037970
# Шаг 5: i = (1007614, 0.14939211) → берём i[0] = 1007614
Результат: новый список только из ID:

Результат: новый список только из ID:

[829, 1811, 1001819, 1037970, 1007614]
"""

artist_by_id.filter(col("id").isin(recommended_artist_ids)).show()

"""
Оценка качества рекомендаций
"""


user_artist_df = raw_user_artist.withColumn(
    "user", split(raw_user_artist["value"], " ").getItem(0).cast(IntegerType())
)

user_artist_df = user_artist_df.withColumn(
    "artist", split(raw_user_artist["value"], " ").getItem(1).cast(IntegerType())
)

user_artist_df = user_artist_df.withColumn(
    "count", split(raw_user_artist["value"], " ").getItem(2).cast(IntegerType())
).drop("value")

user_artist_df.show()

"""def srea_uder_curve(positive_data, 
                    b_all_artist_IDs,
                    predict_function):
    all_data = user_artist_df
"""
