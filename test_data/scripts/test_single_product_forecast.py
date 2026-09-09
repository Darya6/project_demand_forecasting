"""
Интеграционный тест: Прогноз для одного товара
Проверяет полный пайплайн с оптимизацией гиперпараметров
"""

import sys
import time
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.data_processor import DataValidator
from modules.features import FeatureEngineer
from modules.model import RandomForestTrainer


def test_single_product_forecast():
    """Тест пайплайна: Прогноз для одного товара"""

    print("Интеграционный тест: Прогноз для одного товара")

    try:
        print("\nШаг 1: Загрузка данных...")
        demo_path = Path(__file__).parent.parent / 'demo' / 'sales_data_120_days.csv'

        if not demo_path.exists():
            print("Файл не найден, создание демо-данных...")
            from generate_demo_data import generate_demo_dataset
            generate_demo_dataset()

        df = pd.read_csv(demo_path)
        print(f"   Загружено {len(df):,} строк")

        print("\nШаг 2: Валидация данных...")
        df_clean = DataValidator().validate_data(df)
        print(f"   Валидировано: {len(df_clean):,} строк")

        print("\nШаг 3: Выбор товара...")
        product_id = df_clean['product_id'].unique()[0]
        product_data = df_clean[df_clean['product_id'] == product_id]
        print(f"   Товар: {product_id}")
        print(f"   Дней для товара: {len(product_data)}")
        print(f"   Средние продажи: {product_data['sales'].mean():.1f} шт/день")

        print("\nШаг 4: Генерация признаков...")
        fe = FeatureEngineer(forecast_type='single_product')
        df_features = fe.prepare_features(df_clean, product_id=product_id)
        X, y = fe.get_training_data(df_features)
        print(f"   Признаков: {X.shape[1]}")
        print(f"   Обучающих примеров: {len(X)}")

        print("\nШаг 5: Оценка ценовой эластичности...")
        elasticity = fe.estimate_price_elasticity(df_features)
        if elasticity is not None:
            print(f"   Эластичность: {elasticity:.3f}")
        else:
            print(f"   Недостаточно данных о цене, используется значение по умолчанию")

        print("\nШаг 6: Обучение с оптимизацией гиперпараметров...")
        trainer = RandomForestTrainer()

        start_time = time.time()
        model_result = trainer.train_model_with_validation(
            X, y, df_features,
            optimize_hyperparams=True
        )
        training_time = time.time() - start_time

        print(f"   Время обучения: {training_time:.2f} сек")
        print(f"   Использована оптимизация: {model_result['used_optimization']}")

        print("\nРезультаты кросс-валидации:")
        for fold in model_result['all_folds_metrics']:
            print(f"   Фолд {fold['fold']}: sMAPE = {fold['smape']:.2f}%, "
                  f"RMSE = {fold['rmse']:.2f}")

        print("\nИтоговые метрики:")
        print(f"   Лучший sMAPE: {model_result['smape']:.2f}%")
        print(f"   Лучший RMSE: {model_result['rmse']:.2f}")
        print(f"   Средний sMAPE: {model_result['mean_smape']:.2f}%")
        print(f"   Средний RMSE: {model_result['mean_rmse']:.2f}")

        print("\nШаг 7: Построение прогноза (7 дней)...")
        last_date = pd.to_datetime(df_clean['date'].max())
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=7
        )

        model = model_result['model']
        forecast = model.predict(X.tail(7))

        print("Дневной прогноз:")
        for date, pred in zip(future_dates, forecast):
            print(f"   {date.strftime('%d.%m.%Y')}: {pred:.0f} шт")

        print("\nСтатистика прогноза:")
        print(f"   Общий объем: {forecast.sum():.0f} шт")
        print(f"   Среднедневной: {forecast.mean():.0f} шт")
        print(f"   Минимум: {forecast.min():.0f} шт")
        print(f"   Максимум: {forecast.max():.0f} шт")

        print("\nПроверка критериев качества:")
        checks = [
            (model_result['mean_smape'] <= 35,
             f"sMAPE <= 35%: {model_result['mean_smape']:.2f}%"),
            (training_time < 180,
             f"Время обучения < 180 сек: {training_time:.2f} сек"),
            (forecast.min() >= 0,
             f"Все прогнозы >= 0: мин = {forecast.min():.0f}"),
            (len(X) >= 30,
             f"Достаточно данных: {len(X)} примеров"),
        ]

        all_passed = True
        for passed, msg in checks:
            status = "[PASSED]" if passed else "[FAILED]"
            print(f"   {status} {msg}")
            if not passed:
                all_passed = False

        if all_passed:
            print("ТЕСТ ПРОЙДЕН УСПЕШНО")
        else:
            print("ТЕСТ ПРОВАЛЕН")

        return all_passed

    except Exception as e:
        print(f"\nТЕСТ ПРОВАЛЕН С ОШИБКОЙ:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_product_forecast()
    sys.exit(0 if success else 1)