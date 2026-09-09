"""
Интеграционный тест: Прогноз общего спроса
Проверяет полный пайплайн от загрузки данных до построения прогноза
"""

import sys
import time
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.data_processor import DataValidator, DataAnalyzer
from modules.features import FeatureEngineer
from modules.model import RandomForestTrainer


def test_overall_demand_forecast():
    """Тест пайплайна: Прогноз общего спроса"""

    print("Интеграционный тест: Прогноз общего спроса")

    try:
        print("\nШаг 1: Загрузка данных...")
        demo_path = Path(__file__).parent.parent / 'demo' / 'sales_data_120_days.csv'

        if not demo_path.exists():
            print(f"Файл не найден: {demo_path}")
            print("Создание демо-данных...")
            from generate_demo_data import generate_demo_dataset
            generate_demo_dataset()

        df = pd.read_csv(demo_path)
        print(f"   Загружено {len(df):,} строк")

        print("\nШаг 2: Валидация данных...")
        validator = DataValidator()
        df_clean = validator.validate_data(df)
        print(f"   Валидировано: {len(df_clean):,} строк")

        print("\nШаг 3: Анализ датасета...")
        analyzer = DataAnalyzer()
        analysis = analyzer.analyze_dataset(df_clean)
        print(f"   Уникальных товаров: {analysis['unique_products']}")
        print(f"   Период: {analysis['date_range']}")
        print(f"   Средние продажи: {analysis['mean_sales']:.1f} шт/день")

        print("\nШаг 4: Агрегация по дням...")
        df_daily = df_clean.groupby('date').agg({'sales': 'sum'}).reset_index()
        print(f"   Дней в датасете: {len(df_daily)}")
        print(f"   Общий объем продаж: {df_daily['sales'].sum():,} шт")

        print("\nШаг 5: Генерация признаков...")
        fe = FeatureEngineer(forecast_type='overall_demand')
        df_features = fe.prepare_features(df_daily, is_training=True)
        X, y = fe.get_training_data(df_features)
        print(f"   Признаков: {X.shape[1]}")
        print(f"   Обучающих примеров: {len(X)}")

        print("\nШаг 6: Обучение модели...")
        trainer = RandomForestTrainer()

        start_time = time.time()
        model_result = trainer.train_model_with_validation(
            X, y, df_features,
            optimize_hyperparams=False
        )
        training_time = time.time() - start_time

        print(f"   Время обучения: {training_time:.2f} сек")
        print(f"   sMAPE: {model_result['mean_smape']:.2f}%")
        print(f"   RMSE: {model_result['mean_rmse']:.2f}")

        print("\nШаг 7: Построение прогноза (7 дней)...")
        last_date = pd.to_datetime(df_daily['date'].max())
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
        print(f"   Волатильность: {forecast.std():.2f} шт")

        print("\nПроверка критериев качества:")
        checks = [
            (model_result['mean_smape'] <= 30,
             f"sMAPE <= 30%: {model_result['mean_smape']:.2f}%"),
            (training_time < 120,
             f"Время обучения < 120 сек: {training_time:.2f} сек"),
            (forecast.min() >= 0,
             f"Все прогнозы >= 0: мин = {forecast.min():.0f}"),
            (len(forecast) == 7,
             f"Сгенерировано 7 дней: {len(forecast)} дней"),
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
    success = test_overall_demand_forecast()
    sys.exit(0 if success else 1)