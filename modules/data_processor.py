"""
Модуль загрузки, валидации и предварительного анализа данных.
Обеспечивает адаптацию названий столбцов, проверку форматов,
агрегацию данных по дням и базовый статистический анализ.
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging

def adapt_dataset_columns(df):
    """Адаптация названий столбцов к стандартным"""
    df_adapted = df.copy()
    df_adapted.columns = df_adapted.columns.str.lower().str.strip()

    column_mapping = {
        # Продажи
        'quantity': 'sales',
        'units': 'sales',
        'volume': 'sales',
        'qty': 'sales',
        'sales_quantity': 'sales',
        'units_sold': 'sales',
        'объем': 'sales',
        'количество': 'sales',

        # Товар
        'sku': 'product_id',
        'item_id': 'product_id',
        'product': 'product_id',
        'артикул': 'product_id',
        'товар': 'product_id',

        # Цена
        'price_unit': 'price',
        'unit_price': 'price',
        'sales_price': 'price',
        'цена': 'price',
        'стоимость': 'price',

        # Дата
        'order_date': 'date',
        'sales_date': 'date',
        'дата': 'date',
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df_adapted.columns and new_col not in df_adapted.columns:
            df_adapted[new_col] = df_adapted[old_col]

    return df_adapted


def aggregate_daily_sales(df):
    """Агрегирует данные по дням и товарам"""
    required_cols = ['date', 'product_id', 'sales']
    if not all(col in df.columns for col in required_cols):
        return df

    aggregation_dict = {'sales': 'sum'}
    if 'price' in df.columns:
        aggregation_dict['price'] = 'mean'

    aggregated = df.groupby(['date', 'product_id']).agg(aggregation_dict).reset_index()
    return aggregated


class DataValidator:
    """Модуль загрузки и валидации данных"""

    def __init__(self):
        self.required_columns = ['date', 'product_id', 'sales']

    def validate_data(self, df):
        """Валидация данных"""
        df = adapt_dataset_columns(df)

        # Проверка обязательных полей
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные поля: {missing}")

        # Проверка на пустые ID товаров
        if df['product_id'].isna().any():
            nan_count = df['product_id'].isna().sum()
            raise ValueError(
                f"Обнаружены пустые идентификаторы товаров!\n\n"
                f"• Количество пустых ID: {nan_count}\n"
                f"• Все идентификаторы товаров должны быть заполнены\n\n"
            )

        # Проверка длины идентификатора товара
        for idx, pid in df['product_id'].items():
            pid_str = str(pid)
            if len(pid_str) > 100:
                display_id = pid_str[:50] + "..." if len(pid_str) > 50 else pid_str
                raise ValueError(
                    f"Слишком длинный идентификатор товара!\n\n"
                    f"• Найден ID: {display_id}\n"
                    f"• Максимальная длина: 100 символов\n"
                )

        # Валидация формата даты
        try:
            df['date'] = pd.to_datetime(
                df['date'],
                dayfirst=True,
                infer_datetime_format=True,
                errors='raise'
            )
        except Exception as e:
            raise ValueError(
                "Некорректный формат даты. Поддерживаемые форматы: DD.MM.YYYY или YYYY-MM-DD."
            )

        # Обработка отрицательных значений продаж
        if (df['sales'] < 0).any():
            df['sales'] = df['sales'].clip(lower=0)

        if pd.api.types.is_float_dtype(df['sales']):
            df['sales'] = df['sales'].round().astype(int)

        # Валидация и форматирование цены
        if 'price' in df.columns:
            try:
                df['price'] = pd.to_numeric(df['price'], errors='coerce')

                invalid_price_mask = df['price'] <= 0
                invalid_count = invalid_price_mask.sum()

                if invalid_count > 0:
                    df.loc[invalid_price_mask, 'price'] = np.nan

                    st.warning(f"""
                    **Внимание: Обнаружены некорректные данные о цене**

                    • Найдено записей с ценой ≤ 0: **{invalid_count}**
                    • **Важно:** Продажи сохранены и будут использованы для прогнозирования
                    • Цена заменена на "неизвестную" для этих записей
                    • Сценарный анализ доступен только по дням с корректной ценой

                    **Рекомендация:** Для точного сценарного анализа загрузите файл 
                    с корректными ценами (все значения > 0).
                    """)

                df['price'] = df['price'].round(2)

            except Exception as e:
                raise ValueError(f"Ошибка обработки столбца 'price': {e}")

        # Агрегация данных
        df = aggregate_daily_sales(df)

        # Проверка достаточности данных
        date_range = df['date'].max() - df['date'].min()
        unique_days = df['date'].nunique()

        if unique_days < 30:
            raise ValueError(f"Недостаточно данных для прогноза! В файле всего {unique_days} уникальных дней. Требуется минимум 30.")

        if date_range.days < 90:
            st.warning(f"Мало данных: всего {date_range.days} дней. "
                       f"Рекомендуется минимум 90 дней (3 месяца) для точного прогноза.")
        elif date_range.days < 180:
            st.info(f"Период данных: {date_range.days} дней. "
                    f"Для учета сезонности рекомендуется 180+ дней.")

        return df


class DataAnalyzer:
    """Анализатор данных"""
    def analyze_dataset(self, df):
        analysis = {
            'total_records': len(df),
            'unique_products': df['product_id'].nunique(),
            'date_range': f"{df['date'].min().strftime('%d.%m.%Y')} - {df['date'].max().strftime('%d.%m.%Y')}",
            'mean_sales': df['sales'].mean(),
            'max_sales': df['sales'].max(),
            'min_sales': df['sales'].min(),
            'median_sales': df['sales'].median(),
            'std_sales': df['sales'].std()
        }

        if 'price' in df.columns:
            valid_prices = df['price'].dropna()
            if len(valid_prices) > 0:
                analysis['mean_price'] = valid_prices.mean()
                analysis['price_range'] = f"{valid_prices.min():.2f} - {valid_prices.max():.2f} ед."
                analysis['valid_price_records'] = len(valid_prices)
                analysis['total_price_records'] = len(df)
            else:
                analysis['price_info'] = "Нет корректных данных о цене для анализа"

        return analysis