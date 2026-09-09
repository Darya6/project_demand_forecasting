# Demand Forecasting System (FMCG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

> Автоматизированная ML-система для прогнозирования спроса и сценарного анализа цен в FMCG-ритейле на базе Random Forest.

---

## Быстрый старт

### Онлайн-демо
[Открыть приложение на Streamlit Cloud](здесь будет ссылка)

### Тестовые данные для загрузки
- **[FMCG_2022_2024.csv](https://github.com/Darya6/project_demand_forecasting/raw/main/test_data/demo/FMCG_2022_2024.csv)** (~45 МБ) 
- **[sales_data_120_days.csv](https://github.com/Darya6/project_demand_forecasting/raw/main/test_data/demo/sales_data_120_days.csv)** (~50 КБ) 

---

## Возможности
- **ML-прогнозирование**: Горизонт 7-30 дней с использованием Random Forest.
- **Сценарный анализ**: Моделирование влияния изменения цены на спрос (расчет эластичности).
- **Адаптивное обучение**: Оптимизация гиперпараметров (GridSearchCV) для конкретных товаров.
- **Data Quality**: Автоматическая валидация, очистка и агрегация данных.
- **Экспорт отчетов**: Генерация результатов в CSV, PDF (с поддержкой кириллицы), Excel, Word, ZIP.

---

## Требования к данным
**Обязательные поля:**
- `date` — дата (YYYY-MM-DD или DD.MM.YYYY).
- `product_id` — ID товара (до 100 символов).
- `sales` — объем продаж (целое число ≥ 0).

**Опционально:**
- `price` — цена (необходима для сценарного анализа).

**Минимум:** 30 дней истории (рекомендуется 90+ для учета сезонности).

---

## Документация проекта
- [**Обзор проекта**](./docs/PROJECT_OVERVIEW.md) — цели, стек технологий и достигнутые результаты.
- [**Спецификация требований**](./docs/REQUIREMENTS.md) — бизнес- и функциональные требования.
- [**Архитектура системы**](./docs/ARCHITECTURE.md) — UML-диаграммы.
- [**Методология тестирования**](./test_data/README.md) — описание тестовых наборов и скриптов.

---

## Локальный запуск
```
git clone https://github.com/ваш-username/project_demand_forecasting.git
cd project_demand_forecasting
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
