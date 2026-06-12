# Импорт библиотеки pandas для работы с таблицами (DataFrame)
import pandas as pd

# Импорт библиотеки numpy для работы с массивами и математическими функциями
import numpy as np

# Импорт Pipeline из sklearn - позволяет объединить несколько шагов обработки в один конвейер
from sklearn.pipeline import Pipeline

# Импорт SimpleImputer - для заполнения пропущенных значений (NaN)
from sklearn.impute import SimpleImputer

# Импорт StandardScaler - для стандартизации данных (приведение к среднему 0 и дисперсии 1)
# Импорт OneHotEncoder - для преобразования категориальных переменных в бинарные столбцы (one-hot кодирование)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Импорт ColumnTransformer - для применения разных преобразований к разным столбцам
from sklearn.compose import ColumnTransformer

# === СОЗДАНИЕ ТЕСТОВЫХ ДАННЫХ ===
# Простой набор данных с пропущенными значениями и категориальной переменной
data = {
    'Feature1': [1, 2, np.nan, 4, 5],      # Числовой столбец с пропуском (np.nan)
    'Feature2': [10, np.nan, 12, 14, 15],  # Числовой столбец с пропуском
    'Category': ['A', 'B', 'A', 'C', 'B']  # Категориальный столбец (текстовые значения)
}

# Преобразуем словарь в DataFrame pandas (таблицу)
df = pd.DataFrame(data)

# === НАСТРОЙКА ОБРАБОТКИ ДЛЯ ЧИСЛОВЫХ СТОЛБЦОВ ===
# Список названий числовых столбцов, которые будем обрабатывать
numeric_features = ['Feature1', 'Feature2']

# Создаем конвейер для числовых данных
numeric_transformer = Pipeline(steps=[
    # Шаг 1: Заполнение пропусков средним значением
    ('imputer', SimpleImputer(strategy='mean')),  # strategy='mean' - заменяет NaN на среднее арифметическое
    # Шаг 2: Стандартизация (масштабирование) данных
    ('scaler', StandardScaler())  # Приводит данные к распределению: среднее=0, стандартное отклонение=1
])

# === НАСТРОЙКА ОБРАБОТКИ ДЛЯ КАТЕГОРИАЛЬНЫХ СТОЛБЦОВ ===
# Список категориальных столбцов
categorical_features = ['Category']

# Создаем конвейер для категориальных данных
categorical_transformer = Pipeline(steps=[
    # Шаг 1: Заполнение пропусков константой 'missing'
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    # strategy='constant' - заполняет пропуски указанным значением
    # fill_value='missing' - вместо NaN будет вставлена строка 'missing'

    # Шаг 2: One-hot кодирование (преобразование категорий в бинарные столбцы)
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    # handle_unknown='ignore' - если встретится неизвестная категория, игнорировать её (не выдавать ошибку)
])

# === ОБЪЕДИНЕНИЕ ПРЕОБРАЗОВАНИЙ ДЛЯ РАЗНЫХ СТОЛБЦОВ ===
# ColumnTransformer позволяет применять разные преобразования к разным колонкам
preprocessor = ColumnTransformer(
    transformers=[
        # ('имя_трансформера', сам_трансформер, список_столбцов)
        ('num', numeric_transformer, numeric_features),  # Для числовых столбцов
        ('cat', categorical_transformer, categorical_features)  # Для категориальных столбцов
    ]
)

# === СОЗДАНИЕ И ЗАПУСК КОНВЕЙЕРА ===
# Создаем основной конвейер, который содержит только предобработку
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)  # Шаг: предобработка данных
])

# Применяем конвейер к данным: fit - обучаем на данных (вычисляем средние и т.д.),
# transform - преобразуем данные
transformed_data = pipeline.fit_transform(df)

# === ПРЕОБРАЗОВАНИЕ РЕЗУЛЬТАТА В DATAFRAME ДЛЯ УДОБНОГО ПРОСМОТРА ===
# Получаем названия столбцов после преобразования

# Берем исходные названия числовых столбцов
"""
pipeline.named_steps['preprocessor'] - получаем preprocessor из пайплайна
.named_transformers_['cat'] - получаем categorical_transformer из preprocessor
.named_steps['onehot'] - получаем OneHotEncoder из categorical_transformer
.get_feature_names_out(categorical_features) - получаем массив названий: ['Category_A', 'Category_B', 'Category_C']
.tolist() - преобразуем массив в список Python: ['Category_A', 'Category_B', 'Category_C']
# numeric_features + ... - объединяем списки: ['Feature1', 'Feature2', 'Category_A', 'Category_B', 'Category_C']
"""
feature_names = numeric_features + \
pipeline.named_steps['preprocessor'] \
.named_transformers_['cat'] \
.named_steps['onehot'] \
.get_feature_names_out(categorical_features).tolist()

# Создаем DataFrame из преобразованных данных с правильными названиями столбцов
transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

# === ВЫВОД РЕЗУЛЬТАТОВ ===
print("Исходные данные:")
print(df)

print("\nПреобразованные данные:")
print(transformed_df)