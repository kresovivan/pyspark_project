
"""Давайте применим ту же концепцию, но сделаем код более надежным, эффективным и гибким. 
Допустим, нам необходимо сгруппировать данные сразу по нескольким столбцам и вычислить несколько агрегаций, 
предварительно оставив данные только по одному магазину. 
Вот как можно это сделать:
"""
import pandas as pd
# Чуть более сложные данные
data = {'Store': ['A', 'B', 'A', 'B', 'A', 'B'],
'Sales': [200, 220, 210, 250, 215, 240],
'Category': ['Electronics', 'Clothing', 'Clothing', 'Electronics', 'Electronics',
'Clothing']}
df = pd.DataFrame(data)
# Группировка по полям Store и Category с вычислением разных агрегаций и фильтром на магазин A
agg_sales = df[df['Store'] == 'A'].groupby(['Store', 'Category']).agg(
    avg_sales=('Sales', 'mean'),
    total_sales=('Sales', 'sum')
).reset_index()
print(agg_sales)


"""
ПОСТРОЧНОЕ ОПИСАНИЕ СКРИПТА:

СТРОКА 1: import pandas as pd
Импорт библиотеки pandas для работы с данными и присвоение ей псевдонима pd

СТРОКА 2: # Чуть более сложные данные
Комментарий, указывающий, что набор данных теперь содержит больше колонок (три вместо двух)

СТРОКИ 3-6: data = {'Store': ['A', 'B', 'A', 'B', 'A', 'B'], 'Sales': [200, 220, 210, 250, 215, 240], 'Category': ['Electronics', 'Clothing', 'Clothing', 'Electronics', 'Electronics', 'Clothing']}
Создание словаря data с тремя ключами:
- 'Store': список из 6 значений - названия магазинов (A, B, A, B, A, B)
- 'Sales': список из 6 целых чисел - суммы продаж (200, 220, 210, 250, 215, 240)
- 'Category': список из 6 значений - категории товаров:
  * Индекс 0 (магазин A) - Electronics
  * Индекс 1 (магазин B) - Clothing
  * Индекс 2 (магазин A) - Clothing
  * Индекс 3 (магазин B) - Electronics
  * Индекс 4 (магазин A) - Electronics
  * Индекс 5 (магазин B) - Clothing

СТРОКА 7: df = pd.DataFrame(data)
Преобразование словаря data в DataFrame pandas - создание таблицы с тремя колонками: Store, Sales, Category

СТРОКА 9: # Группировка по полям Store и Category с вычислением разных агрегаций и фильтром на магазин A
Комментарий, описывающий следующую операцию: 
- Фильтрация данных только для магазина A
- Группировка по двум полям (Store и Category)
- Вычисление двух различных агрегированных метрик (среднее и сумма)

СТРОКИ 10-14: agg_sales = df[df['Store'] == 'A'].groupby(['Store', 'Category']).agg(
    avg_sales=('Sales', 'mean'),
    total_sales=('Sales', 'sum')
).reset_index()
Цепочка операций:

1. df[df['Store'] == 'A'] - фильтрация DataFrame: оставляем только строки, где значение в колонке 'Store' равно 'A'
   После фильтрации остаются строки с индексами 0, 2, 4:
   - Индекс 0: Store=A, Sales=200, Category=Electronics
   - Индекс 2: Store=A, Sales=210, Category=Clothing
   - Индекс 4: Store=A, Sales=215, Category=Electronics

2. .groupby(['Store', 'Category']) - группировка отфильтрованных данных по двум колонкам:
   - Группа 1: Store='A', Category='Electronics' (индексы 0 и 4)
   - Группа 2: Store='A', Category='Clothing' (индекс 2)

3. .agg(avg_sales=('Sales', 'mean'), total_sales=('Sales', 'sum')) - агрегация с созданием новых колонок:
   - avg_sales: вычисление среднего арифметического колонки 'Sales' для каждой группы
   - total_sales: вычисление суммы значений колонки 'Sales' для каждой группы
   
   Результаты агрегации:
   - Electronics: avg_sales = (200 + 215) / 2 = 207.5, total_sales = 200 + 215 = 415
   - Clothing: avg_sales = 210, total_sales = 210

4. .reset_index() - преобразование индексов (Store и Category) обратно в обычные колонки
   Без reset_index() Store и Category были бы частью индекса, после - становятся обычными колонками DataFrame

Результат сохраняется в переменную agg_sales - DataFrame с колонками: Store, Category, avg_sales, total_sales

СТРОКА 15: print(agg_sales)
Вывод на экран полученного DataFrame agg_sales:
  Store      Category  avg_sales  total_sales
0     A     Clothing      210.0          210
1     A  Electronics      207.5          415
"""