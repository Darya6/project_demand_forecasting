"""
Модуль инженерии признаков. Создает 20+ признаков для временных рядов.
Оценивает ценовую эластичность спроса для сценарного анализа.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


class FeatureEngineer:
    """Инженер признаков для временных рядов"""

    def __init__(self, forecast_type='single_product'):
        self.forecast_type = forecast_type
        self.feature_columns = []
        self.price_elasticity = None

    def prepare_features(self, df, product_id=None, is_training=True):
        """Подготовка признаков для моделирования"""
        if self.forecast_type == 'single_product' and product_id:
            df = df[df['product_id'] == product_id].copy()

        df = df.copy().sort_values('date')

        # Базовые временные признаки
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter

        # Сезонные признаки
        df['is_holiday_peak'] = df['month'].isin([11, 12]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([1, 2]).astype(int)

        # Лаговые признаки
        for lag in [1, 2, 7, 14]:
            df[f'lag_{lag}'] = df['sales'].shift(lag)

        # Скользящие статистики
        windows = [7, 14]
        for window in windows:
            df[f'rolling_mean_{window}'] = df['sales'].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['sales'].rolling(window, min_periods=1).std().fillna(0)

        # Динамические признаки
        if 'lag_1' in df.columns:
            df['momentum'] = df['sales'] - df['lag_1']
            df['momentum_pct'] = (df['sales'] - df['lag_1']) / (df['lag_1'] + 1e-5)

        # Ценовые признаки
        if 'price' in df.columns:
            for lag in [1, 2, 7]:
                df[f'price_lag_{lag}'] = df['price'].shift(lag)

            if 'price_lag_1' in df.columns:
                df['price_change'] = df['price'] - df['price_lag_1']
                df['price_change_pct'] = (df['price'] - df['price_lag_1']) / (df['price_lag_1'] + 1e-5)

            df['price_normalized'] = df['price'] / df['price'].rolling(30, min_periods=1).mean()

        if is_training:
            df = df.dropna()
        else:
            df = df.fillna(0)

        exclude_cols = ['date', 'sales', 'product_id']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = [col for col in self.feature_columns if pd.api.types.is_numeric_dtype(df[col])]

        return df

    def estimate_price_elasticity(self, df):
        """Оценка эластичности спроса по цене"""
        if 'price' not in df.columns:
            return None

        df_filtered = df[(df['sales'] > 0) & (df['price'] > 0)].copy()

        if len(df_filtered) < 10:
            return None

        df_filtered['log_sales'] = np.log(df_filtered['sales'])
        df_filtered['log_price'] = np.log(df_filtered['price'])

        X = df_filtered[['log_price']]
        y = df_filtered['log_sales']

        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            elasticity = model.params['log_price']
            return elasticity
        except:
            return -0.5

    def get_training_data(self, df):
        """Получение данных для обучения"""
        X = df[self.feature_columns]
        y = df['sales']
        return X, y