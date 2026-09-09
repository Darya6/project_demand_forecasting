# Тестовые данные

## Демо-датасеты (`demo/`)

### `FMCG_2022_2024.csv` 
- **Источник:** Kaggle ([FMCG Daily Sales Data](https://www.kaggle.com/datasets/beatafaron/fmcg-daily-sales-data-to-2022-2024))
- **Размер:** ~45 МБ, 1.2 млн записей
- **Товары:** 150+ артикулов
- **Период:** 2022-2024 (3 года)

### `sales_data_120_days.csv`
- **Тип:** Синтетические данные
- **Размер:** ~50 КБ, 360 записей
- **Товары:** 3 артикулов
- **Период:** 120 дней

## Тестовые файлы (`tests/`)

Намеренно некорректные данные для проверки валидации

**Подробнее:** [tests/README.md](tests/README.md)

## Скрипты (`scripts/`)

### Генерация данных
```
python scripts/generate_demo_data.py
```

### Валидация данных
```
python scripts/test_3_validation.py 
```

### Прогноз для одного товара
```
python scripts/test_single_product_forecast.py
```

### Общий прогноз спроса
```
python scripts/test_2_overall_demand.py
```

## Быстрый старт
- Скачайте demo/FMCG_2022_2024.csv
- Загрузите в приложение
- Постройте прогноз

### Запуск всех тестов:
``` 
python scripts/test_3_validation.py
```

## Формат данных
Обязательные поля:

- date - дата (YYYY-MM-DD, DD.MM.YYYY)
- product_id - ID товара (≤ 100 символов)
- sales - объем продаж (целое ≥ 0)

Опционально:
-price - цена (> 0, для сценарного анализа)

Минимум: 30 дней данных (рекомендуется 90+)

## При проблемах:
- Проверьте формат данных
- Запустите python scripts/test_3_validation.py
- Изучите примеры в demo/