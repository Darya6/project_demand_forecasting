"""
Модуль машинного обучения. Random Forest с адаптивной стратегией обучения,
оптимизацией гиперпараметров и временной кросс-валидацией.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import streamlit as st


def calculate_symmetric_mape(y_true, y_pred):
    """Расчет симметричного MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[non_zero] - y_true[non_zero]) / denominator[non_zero])


def calculate_rmse(y_true, y_pred):
    """Расчет RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


class RandomForestTrainer:
    """Random Forest для временных рядов"""

    def optimize_hyperparameters(self, X, y):
        """Оптимизация гиперпараметров с помощью GridSearchCV"""
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [6, 8, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_samples': [0.8, 1.0]
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=1)

        tscv = TimeSeriesSplit(n_splits=3)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,
            error_score='raise'
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_

    def train_model_with_validation(self, X, y, df_features=None, optimize_hyperparams=True):
        """Обучение модели с валидацией"""
        if optimize_hyperparams and len(X) > 20:
            try:
                model, best_params = self.optimize_hyperparameters(X, y)
            except Exception as e:
                st.warning(f"Ошибка оптимизации: {e}. Переход на базовые параметры.")
                best_params = {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'random_state': 42,
                    'max_samples': 0.8,
                    'n_jobs': 1
                }
                model = RandomForestRegressor(**best_params)
                model.fit(X, y)
        else:
            best_params = {
                'n_estimators': 100,
                'max_depth': 8,
                'random_state': 42,
                'max_samples': 0.8,
                'n_jobs': 1
            }
            model = RandomForestRegressor(**best_params)
            model.fit(X, y)

        # Кросс-валидация для финальной оценки точности
        tscv = TimeSeriesSplit(n_splits=3, test_size=min(30, len(X) // 5))
        folds_metrics = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            m = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                max_samples=0.8,
                n_jobs=1
            )
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)

            folds_metrics.append({
                'fold': fold + 1,
                'smape': calculate_symmetric_mape(y_te, preds),
                'rmse': calculate_rmse(y_te, preds)
            })

        return {
            'model': model,
            'smape': folds_metrics[-1]['smape'],
            'rmse': folds_metrics[-1]['rmse'],
            'mean_smape': np.mean([f['smape'] for f in folds_metrics]),
            'mean_rmse': np.mean([f['rmse'] for f in folds_metrics]),
            'all_folds_metrics': folds_metrics,
            'used_optimization': optimize_hyperparams
        }