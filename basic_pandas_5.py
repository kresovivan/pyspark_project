import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Создаем простой набор данных
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
sales = [100, 120, np.nan, 140, 160, 150, np.nan, 200, 180, 190,
 210, 205, 215, np.nan, 230, 240, 235, 245, 250, 260,
 255, np.nan, 270, 275, 280, 285, 290, 295, 300, 310]
categories = ['A', 'B', 'C'] * 10
df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Category': categories})
# Отображение информации об исходном датафрейме
print("Информация об исходном датафрейме:")
print(df.info())
print("\nИспользованная память исходным датафреймом:")
print(df.memory_usage(deep=True))
# Подставляем пропущенные значения с помощью прямого заполнения
df['Sales_Filled'] = df['Sales'].ffill()
# Оптимизируем типы данных
df['Sales'] = pd.to_numeric(df['Sales'], downcast='float')
df['Sales_Filled'] = pd.to_numeric(df['Sales_Filled'], downcast='float')
df['Category'] = df['Category'].astype('category')
# Вычисляем различные метрики
df['Rolling_Avg_3d'] = df['Sales_Filled'].rolling(window=3).mean()

df['Rolling_Avg_7d'] = df['Sales_Filled'].rolling(window=7).mean()
df['Pct_Change'] = df['Sales_Filled'].pct_change()
df['Cumulative_Sum'] = df['Sales_Filled'].cumsum()
# Отображаем информацию после оптимизации
print("\nИнформация об оптимизированном датафрейме:")
print(df.info())
print("\nИспользованная память после оптимизации:")
print(df.memory_usage(deep=True))
# Вычисляем показатели по категориям
category_stats = df.groupby('Category')['Sales_Filled'].agg(['mean', 'median', 'std'])
print("\nПоказатели по категориям:")
print(category_stats)
# Визуализируем данные
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Исходные продажи', alpha=0.7)
plt.plot(df['Date'], df['Sales_Filled'], label='Продажи (заполненные)')
plt.plot(df['Date'], df['Rolling_Avg_3d'], label='3-дневное скользящее среднее')
plt.plot(df['Date'], df['Rolling_Avg_7d'], label='7-дневное скользящее среднее')
plt.title('Ежедневные продажи и скользящие средние')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Выводим итоговый датафрейм
print("\nИтоговый датафрейм:")
print(df.head())

"""
ПОСТРОЧНОЕ ОПИСАНИЕ СКРИПТА:

    СТРОКИ 1-3: Импорт библиотек
    - import pandas as pd - импорт библиотеки pandas для работы с данными (DataFrame, серии)
    - import numpy as np - импорт numpy для математических операций и работы с пропусками (np.nan)
    - import matplotlib.pyplot as plt - импорт pyplot для построения графиков и визуализации

    СТРОКА 5: dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    Создание временного ряда: 30 дат с 1 января 2023 года с ежедневной частотой (freq='D')

    СТРОКИ 6-7: sales = [100, 120, np.nan, ... , 310]
    Создание списка продаж из 30 значений. np.nan - это пропущенные значения (None/NaN)

    СТРОКА 8: categories = ['A', 'B', 'C'] * 10
    Создание списка категорий: повторение ['A','B','C'] 10 раз, итого 30 элементов (категории для каждого дня)

    СТРОКА 9: df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Category': categories})
    Создание DataFrame из словаря: три колонки - Date (дата), Sales (продажи с пропусками), Category (категория)

    СТРОКИ 11-14: Вывод информации об исходном DataFrame
    - print("Информация...") - вывод заголовка
    - print(df.info()) - вывод информации о типах данных, количестве не-NaN значений, использовании памяти
    - print("\nИспользованная память...") - вывод заголовка
    - print(df.memory_usage(deep=True)) - вывод детальной информации о памяти, занимаемой каждой колонкой

    СТРОКА 17: df['Sales_Filled'] = df['Sales'].ffill()
    Создание новой колонки 'Sales_Filled' - заполнение пропусков (NaN) методом forward fill (предыдущим значением)

    СТРОКИ 19-21: Оптимизация типов данных
    - df['Sales'] = pd.to_numeric(df['Sales'], downcast='float') - преобразование в float с понижением точности (экономия памяти)
    - df['Sales_Filled'] = pd.to_numeric(df['Sales_Filled'], downcast='float') - аналогично для заполненной колонки
    - df['Category'] = df['Category'].astype('category') - преобразование категориальной колонки в тип 'category' (экономия памяти)

    СТРОКИ 23-27: Вычисление метрик
    - df['Rolling_Avg_3d'] = df['Sales_Filled'].rolling(window=3).mean() - скользящее среднее за 3 дня
    - df['Rolling_Avg_7d'] = df['Sales_Filled'].rolling(window=7).mean() - скользящее среднее за 7 дней
    - df['Pct_Change'] = df['Sales_Filled'].pct_change() - процентное изменение от предыдущего значения
    - df['Cumulative_Sum'] = df['Sales_Filled'].cumsum() - накопительная сумма (кумулятивная)

    СТРОКИ 29-34: Вывод информации после оптимизации
    (аналогично строкам 11-14, но для оптимизированного DataFrame)

    СТРОКИ 36-38: Агрегация по категориям
    - category_stats = df.groupby('Category')['Sales_Filled'].agg(['mean', 'median', 'std']) - группировка по категориям, вычисление для каждой: среднего, медианы, стандартного отклонения
    - print("Показатели по категориям:") и print(category_stats) - вывод результатов

    СТРОКИ 40-51: Визуализация
    - plt.figure(figsize=(12, 6)) - создание графика размером 12x6 дюймов
    - plt.plot(df['Date'], df['Sales'], label='Исходные продажи', alpha=0.7) - линия исходных продаж с прозрачностью 0.7
    - plt.plot(df['Date'], df['Sales_Filled'], label='Продажи (заполненные)') - линия заполненных продаж
    - plt.plot(df['Date'], df['Rolling_Avg_3d'], label='3-дневное скользящее среднее') - линия среднего за 3 дня
    - plt.plot(df['Date'], df['Rolling_Avg_7d'], label='7-дневное скользящее среднее') - линия среднего за 7 дней
    - plt.title('Ежедневные продажи и скользящие средние') - заголовок графика
    - plt.xlabel('Дата') - подпись оси X
    - plt.ylabel('Продажи') - подпись оси Y
    - plt.legend() - отображение легенды
    - plt.xticks(rotation=45) - поворот подписей оси X на 45 градусов
    - plt.tight_layout() - автоматическая настройка отступов
    - plt.show() - отображение графика

    СТРОКИ 53-55: Вывод итогового DataFrame
    - print("\nИтоговый датафрейм:") - вывод заголовка
    - print(df.head()) - вывод первых 5 строк DataFrame
    """