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