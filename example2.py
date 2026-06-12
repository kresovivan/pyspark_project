import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Создаем простой набор данных, на этот раз с помощью списков
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
sales = [100, 120, np.nan, 140, 160, 150, np.nan, 200, 180, 190,
210, 205, 215, np.nan, 230, 240, 235, 245, 250, 260,
255, np.nan, 270, 275, 280, 285, 290, 295, 300, 310]
categories = ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C',
'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A',
'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B']

df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Category': categories})

# Выводим первые строки
print("Исходный датафрейм:")
print(df.head())
print("\nИнформация о датафрейме:")
print(df.info())

# Подставляем пропущенные значения с помощью прямого заполнения
df['Sales_Filled'] = df['Sales'].ffill()  # <--- ЭТО ЕДИНСТВЕННОЕ ИСПРАВЛЕНИЕ

# Рассчитываем скользящие средние с разными интервалами
df['Rolling_Avg_3d'] = df['Sales_Filled'].rolling(window=3).mean()
df['Rolling_Avg_7d'] = df['Sales_Filled'].rolling(window=7).mean()

# Группируем по столбцу Category и вычисляем разные статистики
category_stats = df.groupby('Category')['Sales_Filled'].agg(['mean', 'median', 'std'])
print("\nСтатистика по категориям:")
print(category_stats)

# Оптимизируем типы данных
df['Sales'] = pd.to_numeric(df['Sales'], downcast='float')
df['Sales_Filled'] = pd.to_numeric(df['Sales_Filled'], downcast='float')
df['Rolling_Avg_3d'] = pd.to_numeric(df['Rolling_Avg_3d'], downcast='float')
df['Rolling_Avg_7d'] = pd.to_numeric(df['Rolling_Avg_7d'], downcast='float')

print("\nИспользование памяти после оптимизации:")
print(df.memory_usage(deep=True))

# Визуализация данных
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
print(df)