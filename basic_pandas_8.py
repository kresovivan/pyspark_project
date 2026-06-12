import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Сгенерированные данные: транзакции с продажами
data = {
 'TransactionID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
 'Store': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
 'SalesAmount': [250, 120, 340, 400, 200, np.nan, 180, 300, 220, 150],
 'Discount': [10, 15, 20, 25, 5, 12, np.nan, 18, 8, 22],
 'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', 
 '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08', 
 '2023-01-09', '2023-01-10']),
 'Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 
 'Home', 'Electronics', 'Home', 'Clothing', 'Electronics']
}
df = pd.DataFrame(data)
# 1. Очистка данных и замена пропущенных значений
imputer = SimpleImputer(strategy='mean')
df[['SalesAmount', 'Discount']] = imputer.fit_transform(df[['SalesAmount', 'Discount']])
# 2. Конструирование признаков
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['NetSales'] = df['SalesAmount'] - df['Discount']
df['DiscountPercentage'] = (df['Discount'] / df['SalesAmount']) * 100
# 3. Расширенная фильтрация
high_value_sales = df[(df['SalesAmount'] > 200) & (df['Store'].isin(['A', 'B']))]
# 4. Агрегация и группировка
agg_sales = df.groupby(['Store', 'Category']).agg(
 TotalSales=('NetSales', 'sum'),
 AvgSales=('NetSales', 'mean'),
 MaxDiscount=('Discount', 'max'),
 SalesCount=('TransactionID', 'count')
).reset_index()
# 5. Анализ временных рядов
daily_sales = df.resample('D', on='Date')['NetSales'].sum().reset_index()
# 6. Нормализация
scaler = StandardScaler()
df['NormalizedSales'] = scaler.fit_transform(df[['SalesAmount']])
# 7. Создание сводной таблицы
category_store_pivot = pd.pivot_table(df, values='NetSales', 
 index='Category', 
 columns='Store', 
 aggfunc='sum', 
 fill_value=0)
# Вывод результатов
print("Исходные данные:")
print(df)
print("\nКрупные продажи в магазинах A и B:")
print(high_value_sales)
print("\nАгрегированные продажи:")
print(agg_sales)
print("\nДневные продажи:")
print(daily_sales)
print("\nСводная таблица по категориям товаров и магазинам:")
print(category_store_pivot)


"""
Нам необходимо произвести очистку данных,
чтобы привести их в консистентный вид, применить фильтры для выделения
интересующих нас срезов данных, а также рассчитать агрегации с целью извлечения выводов.


Что происходит в этом коде?
1. Загрузка и предварительная обработка данных:
• генерируем набор данных с суммами транзакций, магазинами, датами продажи и категориями товаров;
• используем класс SimpleImputer для замены пропущенных значений 
на средние в столбцах SalesAmount и Discount.
2. Конструирование признаков:
• извлекаем день недели из столбца с датами;
• создаем столбец NetSales, вычитая значения в  столбце Discount из 
значений в столбце SalesAmount;
• создаем столбец DiscountPercentage с рассчитанным процентом скидки по транзакции.
3. Расширенная фильтрация:
• оставляем в  наборе данных только транзакции с  суммой, превышающей $200, по магазинам A и B, применяя для этого метод isin().
4. Агрегация и группировка:
• группируем данные по столбцам Store и  Category для получения 
укрупненной информации о продажах;
• рассчитываем разные агрегации по столбцам: общие продажи, средние продажи, максимальную скидку и количество транзакций.
5. Анализ временных рядов:
• используем метод resample() для передискретизации временных 
данных по дням, демонстрируя возможности анализа временных 
рядов.
6. Нормализация:
• применяем класс StandardScaler для стандартизации данных в столбце SalesAmount – это зачастую требуется в процессе подготовки данных для моделей машинного обучения.
7. Создание сводной таблицы:
• строим сводную таблицу с укрупненными суммами продаж на пересечении категорий товаров и  магазинов для компактного вывода 
информации.
"""