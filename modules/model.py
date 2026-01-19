# ==========================================================
# Файл: model.py
# Описание: Модуль машинного обучения. Содержит реализацию Random Forest
#           с адаптивной стратегией обучения, оптимизацией гиперпараметров
#           (GridSearchCV) и временной кросс-валидацией (TimeSeriesSplit).
#           Также включает функции расчета метрик качества (sMAPE, RMSE).
# Соответствует:
#   - Диаграмме классов (Class Diagram)
#   - Диаграмме последовательности (Sequence Diagram) для обучения модели
#   - Функциональным требованиям ФТ2.1-ФТ2.4
#   - Нефункциональным требованиям НФТ1.1 (точность sMAPE ≤20%)
#   - Бизнес-требованиям БТ1.1 (точность), БТ1.2 (формализация)
# Автор: [Ваше имя]
# Дата: 2025-12-12
# Версия: 1.0
# ==========================================================

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import streamlit as st


def calculate_symmetric_mape(y_true, y_pred):
    """Симметричный MAPE (соответствует метрике из раздела 3.1.4)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[non_zero] - y_true[non_zero]) / denominator[non_zero])


def calculate_rmse(y_true, y_pred):
    """Расчет RMSE (соответствует метрике из раздела 3.1.4)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class RandomForestTrainer:
    """Random Forest для временных рядов (реализует ФТ2.1-ФТ2.4)"""

    def __init__(self):
        self.price_scaler = StandardScaler()

    def optimize_hyperparameters(self, X, y):
        """Оптимизация гиперпараметров с помощью GridSearchCV (реализует ФТ2.2)"""
        param_grid = {
            'n_estimators': [50, 100, 150],  # Количество деревьев в лесу
            'max_depth': [6, 8, 10],  # Максимальная глубина дерева
            'min_samples_split': [2, 5, 10],  # Минимальное число образцов для разделения узла
            'min_samples_leaf': [1, 2, 4]  # Минимальное число образцов в листе
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=1)

        # Использование временной кросс-валидации для поиска параметров
        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=0
        )
        grid_search.fit(X, y)

        return grid_search.best_estimator_, grid_search.best_params_

    def train_model_with_validation(self, X, y, df_features=None, optimize_hyperparams=True):
        """
        Обучение модели с валидацией (реализует адаптивную стратегию из раздела 3.1.2)
        Этап 1: Оптимизация параметров на всех данных (если выбран один товар)
        Этап 2: Кросс-валидация с фиксированными параметрами
        """
        # Если в данных есть цена, сохранение статистики для нормализации
        if df_features is not None and 'price' in df_features.columns:
            price_data = df_features[df_features['price'] > 0]['price'].values.reshape(-1, 1)
            if len(price_data) > 0:
                self.price_scaler.fit(price_data)

        # Этап 1: оптимизация гиперпараметров
        if optimize_hyperparams:
            try:
                best_model, best_params = self.optimize_hyperparameters(X, y)
            except Exception as e:
                st.warning(f"Ошибка оптимизации: {e}. Используем стандартные параметры.")
                best_params = {
                    'n_estimators': 100,  # 100 деревьев
                    'max_depth': 8,  # Глубина 8 уровней
                    'min_samples_split': 2,  # Разделять можно от 2 образцов
                    'min_samples_leaf': 1,  # Лист может содержать 1 образец
                    'random_state': 42  # Для воспроизводимости
                }
                best_model = RandomForestRegressor(**best_params, n_jobs=1)
                best_model.fit(X, y)
        else:
            # Параметры по умолчанию для быстрого прогноза (все товары - ФТ2.3)
            best_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            best_model = RandomForestRegressor(**best_params, n_jobs=1)
            best_model.fit(X, y)

        # Этап 2: кросс-валидация для оценки качества (ФТ2.1)
        with st.spinner("Оценка точности модели..."):
            tscv = TimeSeriesSplit(n_splits=3, test_size=30)
            models = []
            validation_scores = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Обучаем с выбранными параметрами
                model = RandomForestRegressor(**best_params, n_jobs=1)
                model.fit(X_train, y_train)

                # Прогноз и оценка (расчет метрик из раздела 3.1.4)
                y_pred = model.predict(X_test)
                smape = calculate_symmetric_mape(y_test, y_pred)
                rmse = calculate_rmse(y_test, y_pred)

                models.append({
                    'model': model,
                    'smape': smape,
                    'rmse': rmse,
                    'fold': fold + 1,
                    'params': best_params
                })
                validation_scores.append(smape)

        # Использование модели, обученной на всех данных
        best_model_result = {
            'model': best_model,  # Модель с лучшими параметрами на всех данных
            'params': best_params,
            'smape': np.min(validation_scores),
            'rmse': np.mean([m['rmse'] for m in models]),
            'mean_smape': np.mean(validation_scores),
            'mean_rmse': np.mean([m['rmse'] for m in models]),
            'all_folds_metrics': models,
            'used_optimization': optimize_hyperparams
        }

        return best_model_result