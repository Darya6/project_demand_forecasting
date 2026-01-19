import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
from datetime import datetime
import time
from modules.export import check_fonts_and_warn

warnings.filterwarnings('ignore')

from modules.data_processor import DataValidator, DataAnalyzer
from modules.features import FeatureEngineer
from modules.model import RandomForestTrainer, calculate_symmetric_mape, calculate_rmse
from modules.export import export_csv, export_pdf, export_word, export_excel, create_zip_file

# ==========================================================
# ОСНОВНОЙ КОД STREAMLIT ПРИЛОЖЕНИЯ
# ==========================================================

# Настройка страницы Streamlit
st.set_page_config(page_title="Прогноз продаж", page_icon="📊", layout="wide")
st.title("Прогнозирование продаж")
st.markdown("---")

# Инициализация состояния шрифтов
if 'fonts_checked' not in st.session_state:
    font_available = check_fonts_and_warn()
    st.session_state.fonts_checked = True
    st.session_state.font_available = font_available

# ==========================================================
# БОКОВАЯ ПАНЕЛЬ НАСТРОЕК
# ==========================================================
with st.sidebar:
    st.header("Настройки")

    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите данные (CSV)", type=['csv'])
    st.caption("Максимальный размер файла: 50 МБ")

    if uploaded_file is not None:
        try:
            # Проверка 1: Пустой файл
            if uploaded_file.size == 0:
                st.error("Файл пуст. Загрузите файл с данными.")
                st.stop()

            # Проверка 2: Размер файла
            MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 МБ
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"Файл слишком большой ({uploaded_file.size / (1024 * 1024):.1f} МБ). Максимум: 50 МБ.")
                st.stop()

            # Информация о файле
            file_size_mb = uploaded_file.size / (1024 * 1024)

            # placeholder для статуса
            status_placeholder = st.empty()

            if file_size_mb > 10:
                status_placeholder.warning(f"Файл большой ({file_size_mb:.1f} МБ). Чтение может занять время...")
            else:
                status_placeholder.info("Чтение файла...")

            start_time = time.time()

            # Чтение файла с индикацией прогресса
            try:
                # Попытка 1: Быстрое чтение
                df = pd.read_csv(uploaded_file)

            except MemoryError:
                # Попытка 2: Чтение по частям для больших файлов
                status_placeholder.info("Файл большой. Чтение может занять время....")

                chunks = []
                chunk_size = 10000  # строк за раз

                # Сбрасывает позицию файла
                uploaded_file.seek(0)

                for i, chunk in enumerate(pd.read_csv(uploaded_file, chunksize=chunk_size)):
                    chunks.append(chunk)

                    # Обновление прогресса каждые 5 чанков
                    if i % 5 == 0:
                        elapsed = time.time() - start_time
                        if elapsed > 60:  # Если уже 60 секунд
                            status_placeholder.error("""
                            Чтение файла занимает слишком много времени.

                            **Рекомендации:**
                            1. Уменьшите размер файла (рекомендуется до 10 МБ)
                            2. Удалите ненужные столбцы
                            3. Разделите файл на части
                            """)
                            st.stop()

                        status_placeholder.info(
                            f"Прочитано {(i + 1) * chunk_size:,} строк... "
                            f"({elapsed:.0f} сек)"
                        )

                df = pd.concat(chunks, ignore_index=True)

            # Проверка результата
            if df.empty:
                status_placeholder.error("Файл не содержит данных.")
                st.stop()

            elapsed_time = time.time() - start_time

            # Обновляет статус на успешное чтение
            status_placeholder.success(f"Файл прочитан успешно!\n")

            time.sleep(2)
            status_placeholder.empty()  # Очистка статуса

            # Проверка расширения файла
            if not uploaded_file.name.lower().endswith('.csv'):
                st.warning("Внимание: Файл не имеет расширения .csv. Убедитесь, что это CSV файл.")


            # Функция загрузки данных с кэшированием
            @st.cache_data
            def load_data(df):
                validator = DataValidator()
                return validator.validate_data(df)


            # Загрузка и валидация данных
            df_clean = load_data(df)

            # Выбор типа анализа
            analysis_type = st.radio("Тип анализа:", ["Один товар", "Все товары вместе"])

            product_id = None
            if analysis_type == "Один товар":
                products = sorted(df_clean['product_id'].unique())
                if len(products) == 0:
                    st.error("В данных не найдено товаров. Проверьте столбец 'product_id'.")
                    st.stop()
                product_id = st.selectbox("Выберите товар:", products)

            # Настройка периода прогноза
            forecast_days = st.slider("Дней для прогноза:", min_value=7, max_value=30, value=14)

            # Проверка на наличие данных о цене для сценарного анализа
            price_data_available = 'price' in df_clean.columns

            # Сценарный анализ
            scenario_active = False
            price_change = 0

            if price_data_available and analysis_type == "Один товар":
                st.divider()
                st.header("Сценарный анализ")

                # Включение сценарного анализ по умолчанию
                scenario_active = st.checkbox("Включить сценарный анализ", value=True)

                if scenario_active:
                    # Текущая средняя цена
                    if product_id:
                        current_price = df_clean[df_clean['product_id'] == product_id]['price'].mean()
                    else:
                        current_price = df_clean['price'].mean()

                    # Отображение текущей цены
                    st.metric("Текущая средняя цена", f"{current_price:.2f} ед.")

                    # Ползунок для изменения цены
                    price_change = st.slider(
                        "Изменение цены, %",
                        min_value=-90,
                        max_value=90,
                        value=-15,
                        step=5,
                        help="Отрицательное значение = снижение цены, положительное = повышение"
                    )

                    # Расчет и отображение новой цены
                    new_price = current_price * (1 + price_change / 100)
                    st.metric("Новая цена", f"{new_price:.2f} ед.", f"{price_change:+.1f}%")

                    # Отображение абсолютного изменения цены
                    price_diff = new_price - current_price
                    st.caption(f"Абсолютное изменение: {price_diff:+.2f} ед.")

            # Кнопка построения прогноза
            run_forecast = st.button("Построить прогноз", type="primary", use_container_width=True)

            # Сохранение настроек в сессии
            st.session_state.df_clean = df_clean
            st.session_state.analysis_type = analysis_type
            st.session_state.product_id = product_id
            st.session_state.forecast_days = forecast_days
            st.session_state.run_forecast = run_forecast
            st.session_state.scenario_active = scenario_active if price_data_available else False
            st.session_state.price_change = price_change if scenario_active else 0

        except pd.errors.EmptyDataError:
            st.error("Ошибка: Загруженный CSV файл пуст или не содержит данных.")
        except Exception as e:
            error_msg = str(e)


            # Определение типа ошибки
            if "Слишком длинный идентификатор товара" in error_msg:
                st.error(f"Проблема с данными: {error_msg}")
                st.write("**Рекомендация:** Укоротите идентификаторы товаров до 100 символов")

            elif "Обнаружены пустые идентификаторы товаров" in error_msg:
                st.error(f"Проблема с данными: {error_msg}")
                st.write("**Рекомендация:** Заполните все пустые значения в столбце 'product_id'")

            else:
                # Для всех остальных ошибок
                st.error(f"Ошибка при обработке файла: {error_msg}")

            st.stop()

# ==========================================================
# ОСНОВНАЯ ОБЛАСТЬ ОТОБРАЖЕНИЯ РЕЗУЛЬТАТОВ
# ==========================================================
if uploaded_file is not None and 'df_clean' in st.session_state:
    df_clean = st.session_state.df_clean
    analyzer = DataAnalyzer()
    analysis = analyzer.analyze_dataset(df_clean)

    # Краткая статистика
    st.subheader("Общая информация о данных")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего записей", f"{analysis['total_records']:,}")
    with col2:
        st.metric("Количество товаров", analysis['unique_products'])
    with col3:
        st.metric("Период данных", analysis['date_range'])

    # Определение, какие данные показывать
    if st.session_state.analysis_type == "Один товар" and st.session_state.product_id:
        historical_data = df_clean[df_clean['product_id'] == st.session_state.product_id]
        title = f"История продаж: {st.session_state.product_id}"
    else:
        historical_data = df_clean.groupby('date').agg({'sales': 'sum'}).reset_index()
        title = "История общего спроса"

    # График исторических продаж
    st.subheader(title)
    fig_sales = go.Figure()
    fig_sales.add_trace(go.Scatter(
        x=historical_data['date'], y=historical_data['sales'],
        mode='lines+markers', name='Продажи',
        line=dict(color='#1f77b4', width=2), marker=dict(size=4),
        hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Продажи: %{y} шт.<extra></extra>'
    ))

    fig_sales.update_layout(
        height=400,
        showlegend=False,
        xaxis=dict(tickformat='%b %Y', tickangle=45),
        yaxis=dict(title="Продажи, шт"),
        margin=dict(l=50, r=50, t=50, b=80)
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    # График цены (только для одного товара)
    if 'price' in historical_data.columns and st.session_state.analysis_type == "Один товар":
        st.subheader(f"История цены: {st.session_state.product_id}")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=historical_data['date'], y=historical_data['price'],
            mode='lines+markers', name='Цена',
            line=dict(color='#2ca02c', width=2), marker=dict(size=4),
            hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Цена: %{y:.2f} ед.<extra></extra>'
        ))

        fig_price.update_layout(
            height=300,
            showlegend=False,
            xaxis=dict(tickformat='%b %Y', tickangle=45),
            yaxis=dict(title="Цена, ед."),
            margin=dict(l=50, r=50, t=30, b=80)
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Статистика цены
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Средняя цена", f"{historical_data['price'].mean():.2f} ед.")
        with col2:
            st.metric("Мин. цена", f"{historical_data['price'].min():.2f} ед.")
        with col3:
            st.metric("Макс. цена", f"{historical_data['price'].max():.2f} ед.")

    # Статистика продаж
    st.subheader("Статистика продаж")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Максимум за день", f"{historical_data['sales'].max():,} шт")
    with col2:
        st.metric("Минимум за день", f"{historical_data['sales'].min():,} шт")
    with col3:
        st.metric("Медиана", f"{historical_data['sales'].median():.0f} шт")
    with col4:
        st.metric("Волатильность", f"{historical_data['sales'].std():.1f} шт")

# ==========================================================
# ПРОГНОЗИРОВАНИЕ
# ==========================================================
if uploaded_file is not None and 'df_clean' in st.session_state and st.session_state.run_forecast:
    with st.spinner("Выполняется расчет прогноза..."):
        try:
            start_time = time.time()

            # Извлечение параметров из сессии
            df_clean = st.session_state.df_clean
            analysis_type = st.session_state.analysis_type
            product_id = st.session_state.product_id
            forecast_days = st.session_state.forecast_days
            scenario_active = st.session_state.get('scenario_active', False)
            price_change = st.session_state.get('price_change', 0)

            # Определение типа прогноза
            forecast_type = "single_product" if analysis_type == "Один товар" else "overall_demand"

            # Подготовка данных
            feature_engineer = FeatureEngineer(forecast_type=forecast_type)
            df_features = feature_engineer.prepare_features(df_clean, product_id, is_training=True)
            X, y = feature_engineer.get_training_data(df_features)

            # Адаптивная оптимизация гиперпараметров
            optimize_hyperparams = (analysis_type == "Один товар")
            optimize_hyperparams = False
            if analysis_type == "Один товар":
                if product_id:
                    product_data = df_clean[df_clean['product_id'] == product_id]
                    optimize_hyperparams = len(product_data) > 100

            if optimize_hyperparams:
                st.info("Режим: Оптимизация гиперпараметров для одного товара")
            else:
                st.info("Режим: Быстрый прогноз для общего спроса")

            # Обучение модели
            trainer = RandomForestTrainer()
            best_model_result = trainer.train_model_with_validation(
                X, y, df_features, optimize_hyperparams=optimize_hyperparams
            )
            model = best_model_result['model']

            # Сохранение метрик качества
            st.session_state.model_metrics = {
                'smape': best_model_result['smape'],
                'rmse': best_model_result['rmse'],
                'mean_smape': best_model_result['mean_smape'],
                'mean_rmse': best_model_result['mean_rmse']
            }

            st.session_state.all_folds_metrics = best_model_result['all_folds_metrics']
            st.session_state.optimization_used = best_model_result['used_optimization']

            # Генерация прогноза
            last_date = df_clean['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

            available_days = min(forecast_days, len(X))
            last_features = X.tail(available_days)

            if forecast_days > available_days:
                repeat_times = int(np.ceil(forecast_days / available_days))
                repeated_features = pd.concat([last_features] * repeat_times, ignore_index=True)
                last_features = repeated_features.head(forecast_days)

            future_predictions = model.predict(last_features)
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'predicted_sales': future_predictions.round().astype(int),
                'confidence_lower': (future_predictions * 0.8).round().astype(int),
                'confidence_upper': (future_predictions * 1.2).round().astype(int)
            })

            # Сценарный анализ
            if scenario_active:
                elasticity = feature_engineer.estimate_price_elasticity(df_features)
                if elasticity is None:
                    elasticity = -0.5

                demand_change_factor = 1 + (elasticity * price_change / 100)

                forecast_df['scenario_sales'] = (forecast_df['predicted_sales'] * demand_change_factor).round().astype(
                    int)
                forecast_df['scenario_lower'] = (forecast_df['confidence_lower'] * demand_change_factor).round().astype(
                    int)
                forecast_df['scenario_upper'] = (forecast_df['confidence_upper'] * demand_change_factor).round().astype(
                    int)

                st.session_state.scenario_params = {
                    'price_change': price_change,
                    'elasticity': elasticity,
                    'demand_change_factor': demand_change_factor
                }

            # Сохранение результатов прогноза
            st.session_state.forecast_results = forecast_df
            st.session_state.forecast_title = f"Прогноз на {forecast_days} дней"
            if product_id:
                st.session_state.forecast_title += f" - {product_id}"
            else:
                st.session_state.forecast_title += " - Все товары"

            end_time = time.time()
            st.session_state.forecast_time = end_time - start_time

            st.success(f"Прогнозирование завершено! Время выполнения: {st.session_state.forecast_time:.1f} сек")
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка при расчете прогноза: {e}")

# ==========================================================
# ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ПРОГНОЗА
# ==========================================================
if 'forecast_results' in st.session_state:
    forecast_df = st.session_state.forecast_results

    st.header("Результаты прогнозирования")
    st.subheader(st.session_state.forecast_title)

    # Информация о сценарном анализе
    if ('scenario_active' in st.session_state and st.session_state.scenario_active and
            'scenario_params' in st.session_state):
        scenario_params = st.session_state.scenario_params
        st.info(f"""
        **Сценарный анализ:**
        - **Изменение цены:** {scenario_params['price_change']:+.1f}%
        - **Оцененная эластичность:** {scenario_params['elasticity']:.2f}
        - **Ожидаемое изменение спроса:** {((scenario_params['demand_change_factor'] - 1) * 100):+.1f}%
        """)

    # Метрики точности модели
    if 'model_metrics' in st.session_state:
        metrics = st.session_state.model_metrics

        if 'optimization_used' in st.session_state:
            if st.session_state.optimization_used:
                st.success("Использована оптимизация гиперпараметров (GridSearchCV)")
            else:
                st.info("Использован быстрый режим (стандартные параметры)")

        st.subheader("Точность модели прогнозирования")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Время прогноза", f"{st.session_state.get('forecast_time', 0):.1f} сек")
        with col2:
            st.metric("Лучший sMAPE", f"{metrics['smape']:.1f}%")
        with col3:
            st.metric("Лучший RMSE", f"{metrics['rmse']:.1f}")
        with col4:
            st.metric("Средний sMAPE", f"{metrics['mean_smape']:.1f}%")
        with col5:
            st.metric("Средний RMSE", f"{metrics['mean_rmse']:.1f}")

        # Детальная информация по фолдам
        if 'all_folds_metrics' in st.session_state:
            with st.expander("Детали по фолдам кросс-валидации"):
                folds_data = []
                for fold_metric in st.session_state.all_folds_metrics:
                    folds_data.append({
                        'Фолд': fold_metric['fold'],
                        'sMAPE (%)': f"{fold_metric['smape']:.1f}",
                        'RMSE': f"{fold_metric['rmse']:.1f}"
                    })
                folds_df = pd.DataFrame(folds_data)
                st.dataframe(folds_df, use_container_width=True)

    # График прогноза
    if 'scenario_sales' in forecast_df.columns:
        # График с двумя сценариями
        fig_forecast = go.Figure()

        # Доверительный интервал базового прогноза
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['confidence_upper'].tolist() + forecast_df['confidence_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(30, 144, 255, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Дов. интервал (базовый)',
            hoverinfo='skip'
        ))

        # Базовый прогноз
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_sales'],
            mode='lines+markers',
            name='Базовый прогноз',
            line=dict(color='blue', width=2),
            marker=dict(size=6, color='blue'),
            hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Базовый: %{y} шт.<extra></extra>'
        ))

        # Доверительный интервал сценария
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['scenario_upper'].tolist() + forecast_df['scenario_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 140, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Дов. интервал (сценарий)',
            hoverinfo='skip'
        ))

        # Сценарный прогноз
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['scenario_sales'],
            mode='lines+markers',
            name='Сценарный прогноз',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6, color='orange'),
            hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Сценарий: %{y} шт.<extra></extra>'
        ))

        fig_forecast.update_layout(
            height=400,
            xaxis=dict(title="Дата", tickformat='%d.%m'),
            yaxis=dict(title="Продажи, шт"),
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Сравнительная таблица
        with st.expander("Детальное сравнение прогнозов"):
            display_data = forecast_df.copy()
            display_data['date'] = display_data['date'].dt.strftime('%d.%m.%Y')
            display_data['Разница (шт)'] = display_data['scenario_sales'] - display_data['predicted_sales']
            display_data['Разница (%)'] = (display_data['Разница (шт)'] / display_data['predicted_sales'] * 100).round(
                2)

            display_data = display_data[['date', 'predicted_sales', 'scenario_sales', 'Разница (шт)', 'Разница (%)']]
            display_data.columns = ['Дата', 'Базовый прогноз', 'Сценарный прогноз', 'Разница (шт)', 'Разница (%)']

            styled_df = display_data.style.format({
                'Базовый прогноз': '{:,}',
                'Сценарный прогноз': '{:,}',
                'Разница (шт)': '{:+,}',
                'Разница (%)': '{:+.2f}'
            })

            st.dataframe(styled_df, use_container_width=True)

    else:
        # Стандартный график без сценария
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['confidence_upper'].tolist() + forecast_df['confidence_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name='Доверительный интервал', hoverinfo='skip'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['predicted_sales'],
            mode='lines+markers', name='Прогноз продаж',
            line=dict(color='red', width=3), marker=dict(size=6),
            hovertemplate='<b>%{x|%d.%m.%Y}</b><br>Прогноз: %{y} шт.<extra></extra>'
        ))
        fig_forecast.update_layout(height=400, xaxis=dict(title="Дата", tickformat='%d.%m'),
                                   yaxis=dict(title="Продажи, шт"), showlegend=True,
                                   margin=dict(l=50, r=50, t=50, b=80)
                                   )
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Сводные метрики прогноза
    st.subheader("Сводные метрики прогноза")

    if 'scenario_sales' in forecast_df.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            base_total = forecast_df['predicted_sales'].sum()
            scenario_total = forecast_df['scenario_sales'].sum()
            st.metric("Общий объем (базовый)", f"{base_total:,} шт")
        with col2:
            change_pct = ((scenario_total / base_total - 1) * 100)
            st.metric("Общий объем (сценарий)", f"{scenario_total:,} шт",
                      f"{change_pct:+.1f}%")
        with col3:
            st.metric("Средние продажи в день",
                      f"{forecast_df['predicted_sales'].mean():.0f} шт",
                      f"{((forecast_df['scenario_sales'].mean() / forecast_df['predicted_sales'].mean() - 1) * 100):+.1f}%")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Общий объем", f"{forecast_df['predicted_sales'].sum():,} шт")
        with col2:
            st.metric("Средние продажи в день", f"{forecast_df['predicted_sales'].mean():.0f} шт")
        with col3:
            variability = (forecast_df['predicted_sales'].std() / forecast_df['predicted_sales'].mean() * 100)
            st.metric("Волатильность", f"{variability:.1f}%")

    # Панель экспорта результатов
    st.subheader("Экспорт результатов")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Скачать CSV", use_container_width=True, type="primary"):
            csv_data = export_csv(forecast_df)
            if csv_data:
                st.download_button(
                    label="Нажмите чтобы скачать CSV",
                    data=csv_data,
                    file_name=f"прогноз_продаж_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="secondary",
                    key="csv_download"
                )

    with col2:
        if st.button("Скачать PDF", use_container_width=True, type="primary"):
            pdf_data = export_pdf(forecast_df, st.session_state.model_metrics)
            if pdf_data:
                st.download_button(
                    label="Нажмите чтобы скачать PDF",
                    data=pdf_data,
                    file_name=f"отчет_прогноз_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="secondary",
                    key="pdf_download"
                )

    with col3:
        if st.button("Скачать Excel", use_container_width=True, type="primary"):
            excel_data = export_excel(forecast_df, st.session_state.model_metrics)
            if excel_data:
                st.download_button(
                    label="Нажмите чтобы скачать Excel",
                    data=excel_data,
                    file_name=f"прогноз_анализ_{timestamp}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True,
                    type="secondary",
                    key="excel_download"
                )

    with col4:
        if st.button("Скачать Word", use_container_width=True, type="primary"):
            word_data = export_word(forecast_df, st.session_state.model_metrics)
            if word_data:
                st.download_button(
                    label="Нажмите чтобы скачать Word",
                    data=word_data,
                    file_name=f"отчет_прогноз_{timestamp}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    type="secondary",
                    key="word_download"
                )

    # Пакетный экспорт
    if st.button("Скачать ВСЕ форматы", use_container_width=True, type="primary"):
        with st.spinner("Создание файлов..."):
            files_dict = {}

            csv_data = export_csv(forecast_df)
            if csv_data:
                files_dict[f"прогноз_продаж_{timestamp}.csv"] = csv_data

            pdf_data = export_pdf(forecast_df, st.session_state.model_metrics)
            if pdf_data:
                files_dict[f"отчет_прогноз_{timestamp}.pdf"] = pdf_data

            excel_data = export_excel(forecast_df, st.session_state.model_metrics)
            if excel_data:
                files_dict[f"прогноз_анализ_{timestamp}.xlsx"] = excel_data

            word_data = export_word(forecast_df, st.session_state.model_metrics)
            if word_data:
                files_dict[f"отчет_прогноз_{timestamp}.docx"] = word_data

            if files_dict:
                if len(files_dict) >= 2:
                    zip_data = create_zip_file(files_dict)
                    st.download_button(
                        label="Нажмите чтобы скачать ZIP архив",
                        data=zip_data,
                        file_name=f"прогноз_продаж_полный_{timestamp}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="secondary",
                        key="zip_download"
                    )
                else:
                    st.warning("Доступные файлы:")
                    for filename, file_data in files_dict.items():
                        if '.csv' in filename:
                            st.download_button(
                                label=f"Скачать {filename}",
                                data=file_data,
                                file_name=filename,
                                mime="text/csv",
                                use_container_width=True,
                                key=f"avail_{filename}"
                            )
                        elif '.pdf' in filename:
                            st.download_button(
                                label=f"Скачать {filename}",
                                data=file_data,
                                file_name=filename,
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"avail_{filename}"
                            )
                        elif '.xlsx' in filename:
                            st.download_button(
                                label=f"Скачать {filename}",
                                data=file_data,
                                file_name=filename,
                                mime="application/vnd.ms-excel",
                                use_container_width=True,
                                key=f"avail_{filename}"
                            )
                        elif '.docx' in filename:
                            st.download_button(
                                label=f"Скачать {filename}",
                                data=file_data,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                use_container_width=True,
                                key=f"avail_{filename}"
                            )
            else:
                st.error("Не удалось создать файлы для экспорта")

# ==========================================================
# ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ
# ==========================================================
elif uploaded_file is None:
    st.info("""
    **Инструкция по использованию:**

    1. **Загрузите CSV файл с историей продаж**
       - Максимальный размер файла: **50 МБ**
       - Рекомендуемый размер: до 10 МБ для быстрой обработки

    2. **Выберите тип анализа:**
       - *Один товар* - прогноз для конкретного товара
       - *Все товары вместе* - общий прогноз спроса

    3. **Настройте прогноз:**
       - Укажите период прогноза (7-30 дней)
       - При анализе одного товара с данными о цене доступен сценарный анализ

    4. **Нажмите "Построить прогноз"**

    **Формат данных:**
    - Обязательные колонки: `date`, `product_id` (не более 100 символов), `sales`
    - Для сценарного анализа: `price` (или `price_unit`, `unit_price`)
    - Поддерживаемые форматы дат: YYYY-MM-DD, DD.MM.YYYY
    - Кодировка файла: UTF-8

    **Если возникли проблемы с загрузкой:**
    1. Проверьте размер файла (должен быть ≤ 50 МБ)
    2. Убедитесь, что файл не поврежден
    3. Попробуйте пересохранить файл в новом CSV формате
    4. Обновите страницу и попробуйте снова
    """)
# ==========================================================
# ФУТЕР ПРИЛОЖЕНИЯ
# ==========================================================
st.markdown("---")
st.caption("Система прогнозирования продаж • Random Forest • Plotly • Сценарный анализ")