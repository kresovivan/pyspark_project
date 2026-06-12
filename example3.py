import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Создаем простой набор данных только с продажами
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
sales = [100, 120, np.nan, 140, 160, 150, np.nan, 200, 180, 190,
         210, 205, 215, np.nan, 230, 240, 235, 245, 250, 260,
         255, np.nan, 270, 275, 280, 285, 290, 295, 300, 310]

df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Подставляем пропущенные значения с помощью прямого заполнения
df['Sales_Filled'] = df['Sales'].ffill() 

# Рассчитываем скользящие средние с разными интервалами
df['Rolling_Avg_3d'] = df['Sales_Filled'].rolling(window=3).mean()
df['Rolling_Avg_7d'] = df['Sales_Filled'].rolling(window=7).mean()
df['Rolling_Avg_14d'] = df['Sales_Filled'].rolling(window=14).mean()

# Вычисляем процентные изменения
df['Pct_Change'] = df['Sales_Filled'].pct_change()

# Рассчитываем накопительную сумму
df['Cumulative_Sum'] = df['Sales_Filled'].cumsum()

# Отображаем результаты
print(df)

# Визуализируем данные
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales_Filled'],    label = 'Filled Sales')
plt.plot(df['Date'], df['Rolling_Avg_3d'],  label = '3-дневное скользящее среднее')
plt.plot(df['Date'], df['Rolling_Avg_7d'],  label = '7-дневное скользящее среднее')
plt.plot(df['Date'], df['Rolling_Avg_14d'], label = '14-дневное скользящее среднее')
plt.title('Ежедневные продажи и скользящие средние')
plt.xlabel('Дата')
plt.ylabel('Продажи')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()