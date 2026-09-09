"""
Генератор демонстрационных данных для тестирования системы
Создает данные продаж с сезонностью, трендом и шумом
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_demo_dataset(
        days=120,
        products=None,
        start_date=None,
        output_file='sales_data_120_days.csv'
):
    """
    Создает CSV с демонстрационными данными

    Args:
        days (int): Количество дней данных (по умолчанию 120)
        products (list): Список товаров (по умолчанию MI-006, MI-007, MI-008)
        start_date (datetime): Начальная дата (по умолчанию 2024-01-01)
        output_file (str): Имя выходного файла

    Returns:
        pd.DataFrame: Сгенерированные данные
    """

    base_path = Path(__file__).parent.parent / 'demo'
    base_path.mkdir(parents=True, exist_ok=True)
    output_path = base_path / output_file

    if start_date is None:
        start_date = datetime(2024, 1, 1)

    if not isinstance(start_date, datetime):
        start_date = datetime.fromisoformat(start_date)

    if products is None:
        products = ['MI-006', 'MI-007', 'MI-008']

    data = []
    np.random.seed(42)

    for i in range(days):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime('%d.%m.%Y')

        for product in products:
            # Базовые параметры товара
            if product == 'MI-006':
                base_sales = 15
                base_price = 5.0
            elif product == 'MI-007':
                base_sales = 25
                base_price = 3.5
            else:  # MI-008
                base_sales = 10
                base_price = 7.0

            # Сезонность (выходные)
            weekend_boost = 1.5 if current_date.weekday() >= 5 else 1.0

            # Тренд (0.5% роста в день)
            trend = 1.0 + (i * 0.005)

            # Случайный шум
            noise = np.random.normal(1.0, 0.1)

            # Итоговые продажи
            sales = max(1, int(base_sales * weekend_boost * trend * noise))

            # Цена с колебаниями ±10%
            price = round(base_price * np.random.uniform(0.9, 1.1), 2)

            data.append({
                'date': date_str,
                'product_id': product,
                'sales': sales,
                'price': price
            })

    df = pd.DataFrame(data)

    # Сохранение
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"Файл создан: {output_path}")
    print(f"   Строк: {len(df):,}")
    print(f"   Период: {df['date'].min()} - {df['date'].max()}")
    print(f"   Товаров: {df['product_id'].nunique()}")
    print(f"   Средние продажи: {df['sales'].mean():.1f} шт/день")

    return df


def generate_demo_with_options(**kwargs):
    """Вспомогательная функция для создания датасетов с опциями"""
    return generate_demo_dataset(**kwargs)


if __name__ == "__main__":
    print("Генерация демонстрационных датасетов...")

    print("\n1. Стандартный датасет (120 дней)...")
    generate_demo_dataset(
        days=120,
        products=['MI-006', 'MI-007', 'MI-008'],
        start_date=datetime(2024, 1, 1),
        output_file='sales_data_120_days.csv'
    )

    print("\n2. Короткий датасет (30 дней)...")
    generate_demo_dataset(
        days=30,
        products=['MI-006'],
        start_date=datetime(2024, 9, 1),
        output_file='sales_data_30_days.csv'
    )

    print("\n3. Датасет (90 дней)...")
    generate_demo_dataset(
        days=90,
        products=['MI-006', 'MI-007', 'MI-008'],
        start_date=datetime(2024, 4, 1),
        output_file='sales_data_90_days.csv'
    )

    print("\n4. Датасет за 2023 год...")
    generate_demo_dataset(
        days=365,
        products=['MI-006', 'MI-007', 'MI-008'],
        start_date=datetime(2023, 1, 1),
        output_file='sales_data_2023_full_year.csv'
    )

    print("Все датасеты созданы.")