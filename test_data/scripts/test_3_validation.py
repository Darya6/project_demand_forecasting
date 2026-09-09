"""
Модульные тесты: Валидация данных
Проверяет обработку некорректных входных данных
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.data_processor import DataValidator


def test_negative_sales():
    """Тест 1: Отрицательные и дробные продажи (с достаточным количеством строк)"""
    print("\nТест 1: Отрицательные и дробные продажи")

    validator = DataValidator()

    # Создает 30 дней данных
    dates = pd.date_range('2024-01-01', periods=30).strftime('%d.%m.%Y')
    sales_values = [10] * 30
    sales_values[0] = -10  # Отрицательное значение
    sales_values[1] = 15.7 # Дробное значение

    df_bad = pd.DataFrame({
        'date': dates,
        'product_id': ['A'] * 30,
        'sales': sales_values
    })

    try:
        df_fixed = validator.validate_data(df_bad)

        has_negatives_after = (df_fixed['sales'] < 0).any()
        all_integers = np.array_equal(df_fixed['sales'], df_fixed['sales'].astype(int))

        checks = [
            (not has_negatives_after, "   Отрицательные значения исправлены на 0"),
            (all_integers, "   Дробные значения округлены до целых"),
        ]

        all_passed = all(check[0] for check in checks)

        for passed, msg in checks:
            status = "[PASSED]" if passed else "[FAILED]"
            print(f"{status} {msg}")

        return all_passed

    except Exception as e:
        print(f"[FAILED] {e}")
        return False


def test_missing_columns():
    """Тест 2: отсутствие обязательных полей"""
    print("\nТест 2: Отсутствие обязательных полей")
    validator = DataValidator()
    df_bad = pd.DataFrame({
        'date': ['01.01.2024', '02.01.2024'],
        'product_id': ['A', 'B']
    })
    try:
        validator.validate_data(df_bad)
        print("[FAILED] Должна была выброситься ошибка")
        return False
    except ValueError as e:
        if "Отсутствуют обязательные поля" in str(e):
            print("[PASSED] Обнаружено отсутствие поля 'sales'")
            return True
        return False


def test_empty_product_id():
    """Тест 3: Пустые идентификаторы товаров"""
    print("\nТест 3: Пустые идентификаторы товаров")
    validator = DataValidator()
    df_bad = pd.DataFrame({
        'date': ['01.01.2024', '02.01.2024'],
        'product_id': ['A', None],
        'sales': [10, 20]
    })
    try:
        validator.validate_data(df_bad)
        return False
    except ValueError as e:
        if "пустые идентификаторы товаров" in str(e).lower():
            print("[PASSED] Обнаружены пустые значения в product_id")
            return True
        return False


def test_long_product_id():
    """Тест 4: Слишком длинные идентификаторы"""
    print("\nТест 4: Слишком длинные идентификаторы товаров")
    validator = DataValidator()
    df_bad = pd.DataFrame({
        'date': ['01.01.2024'],
        'product_id': ['A' * 150],
        'sales': [10]
    })
    try:
        validator.validate_data(df_bad)
        return False
    except ValueError as e:
        if "длинный идентификатор" in str(e).lower():
            print("[PASSED] Обнаружен ID длиннее 100 символов")
            return True
        return False


def test_bad_date_format():
    """Тест 5: Некорректный формат даты"""
    print("\nТест 5: Некорректный формат даты")
    validator = DataValidator()
    df_bad = pd.DataFrame({
        'date': ['2024/13/45'],
        'product_id': ['A'],
        'sales': [10]
    })
    try:
        validator.validate_data(df_bad)
        return False
    except ValueError as e:
        if "формат даты" in str(e).lower():
            print("[PASSED] Обнаружена ошибка формата даты")
            return True
        return False


def test_insufficient_data():
    """Тест 6: Недостаточно данных (< 30 дней)"""
    print("\nТест 6: Недостаточно данных для прогноза")
    validator = DataValidator()
    df_bad = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10).strftime('%d.%m.%Y'),
        'product_id': ['A'] * 10,
        'sales': [10] * 10
    })
    try:
        validator.validate_data(df_bad)
        return False
    except ValueError as e:
        if "недостаточно данных" in str(e).lower():
            print("[PASSED] Обнаружено недостаточное количество данных")
            return True
        return False


if __name__ == "__main__":
    print("Запуск комплексного тестирования валидации...")
    tests = [
        test_negative_sales,
        test_missing_columns,
        test_empty_product_id,
        test_long_product_id,
        test_bad_date_format,
        test_insufficient_data
    ]
    results = [t() for t in tests]
    print(f"\nИтог: {sum(results)}/{len(results)} тестов пройдено")
    sys.exit(0 if all(results) else 1)