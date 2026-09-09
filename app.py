"""
Главный модуль системы прогнозирования спроса.
Реализует полный цикл: загрузка, валидация, ML-прогноз,
сценарный анализ, визуализация и экспорт.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
from datetime import datetime

from modules.data_processor import DataValidator, DataAnalyzer
from modules.features import FeatureEngineer
from modules.model import RandomForestTrainer
from modules.export import (
    export_csv, export_pdf, export_word,
    export_excel, create_zip_file, check_fonts_and_warn, export_model
)

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Прогноз продаж", page_icon="📊", layout="wide")

# Инициализация состояний сессии для сохранения данных при обновлении страницы
if 'df_clean' not in st.session_state: st.session_state.df_clean = None
if 'forecast_done' not in st.session_state: st.session_state.forecast_done = False
if 'forecast_results' not in st.session_state: st.session_state.forecast_results = None
if 'model_metrics' not in st.session_state: st.session_state.model_metrics = None

# Проверка шрифтов для корректного отображения кириллицы в PDF
if 'fonts_checked' not in st.session_state:
    st.session_state.font_available = check_fonts_and_warn()
    st.session_state.fonts_checked = True


# Функция кэширования для исключения повторного чтения тяжелых файлов
@st.cache_data
def get_validated_data(file_input):
    """Загрузка и валидация данных"""
    df = pd.read_csv(file_input)
    validator = DataValidator()
    return validator.validate_data(df)


st.title("Система прогнозирования спроса")
st.markdown("---")

# Боковая панель настроек
with st.sidebar:
    st.header("Загрузка данных")
    source_mode = st.radio("Источник данных:", ["Загрузить свой CSV", "Использовать демо-набор"])

    input_data = None

    if source_mode == "Загрузить свой CSV":
        uploaded_file = st.file_uploader("Выберите файл", type=['csv'], help="Максимальный размер 50 МБ")
        if uploaded_file:
            input_data = uploaded_file
    else:
        demo_choice = st.selectbox(
            "Выберите сценарий:",
            ["Масштабный (1.2 млн строк)", "Минимальный (120 дней)"]
        )
        if st.button("Загрузить демо-данные", use_container_width=True):
            input_data = (
                "test_data/demo/FMCG_2022_2024.csv"
                if "Масштабный" in demo_choice
                else "test_data/demo/sales_data_120_days.csv"
            )

    # Валидация при изменении входных данных
    if input_data is not None:
        try:
            with st.spinner("Валидация и очистка данных..."):
                st.session_state.df_clean = get_validated_data(input_data)
                st.sidebar.success("Данные успешно подготовлены")
                st.session_state.forecast_done = False  # Сброс старых результатов
        except Exception as e:
            st.error(f"Ошибка данных: {e}")

    # Блок настройки параметров прогноза
    if st.session_state.df_clean is not None:
        st.divider()
        st.header("Параметры прогноза")
        df_clean = st.session_state.df_clean

        analysis_type = st.radio("Объект анализа:", ["Один товар", "Все товары вместе"])

        product_id = None
        if analysis_type == "Один товар":
            product_id = st.selectbox("Выберите артикул:", sorted(df_clean['product_id'].unique()))

        forecast_days = st.slider("Горизонт (дней):", 7, 30, 14)

        # Логика сценарного анализа
        scenario_active = False
        price_change = 0
        if 'price' in df_clean.columns and analysis_type == "Один товар":
            st.divider()
            scenario_active = st.checkbox("Включить сценарный анализ")
            if scenario_active:
                price_change = st.slider("Изменение цены, %", -90, 90, -15)

        run_forecast = st.button("Построить прогноз", type="primary", use_container_width=True)


# Основная область
if st.session_state.df_clean is not None:
    df_clean = st.session_state.df_clean
    analyzer = DataAnalyzer()
    stats = analyzer.analyze_dataset(df_clean)

    # Вывод общей статистики
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Всего записей", f"{stats['total_records']:,}")
    col_stat2.metric("Количество уникальных товаров", stats['unique_products'])
    col_stat3.metric("Период данных", stats['date_range'])

    # Визуализация истории
    if analysis_type == "Один товар":
        hist_data = df_clean[df_clean['product_id'] == product_id]
        hist_title = f"История продаж товара: {product_id}"
    else:
        hist_data = df_clean.groupby('date').agg({'sales': 'sum'}).reset_index()
        hist_title = "История совокупного спроса"

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist_data['date'], y=hist_data['sales'], name="Факт", line=dict(color='#1f77b4')))
    fig_hist.update_layout(title=hist_title, height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # Запуск ML-пайплайна
    if 'run_forecast' in locals() and run_forecast:
        with st.status("Выполнение расчетов. Пожалуйста, подождите...", expanded=True) as status:

            if analysis_type == "Все товары вместе":
                df_for_model = df_clean.groupby('date').agg({'sales': 'sum', 'price': 'mean'}).reset_index()
                p_id_to_fe = None
            else:
                df_for_model = df_clean
                p_id_to_fe = product_id

            fe = FeatureEngineer(forecast_type='single_product' if product_id else 'overall_demand')
            df_features = fe.prepare_features(df_for_model, p_id_to_fe)
            X, y = fe.get_training_data(df_features)

            # Пояснение для пользователя
            st.caption("Время обучения зависит от объема данных и может занимать до нескольких минут.")

            trainer = RandomForestTrainer()
            optimize = (analysis_type == "Один товар" and len(df_features) > 40)
            res = trainer.train_model_with_validation(X, y, df_features, optimize_hyperparams=optimize)

            st.write("Формирование прогнозных значений и доверительных интервалов...")
            model = res['model']
            preds = model.predict(X.tail(forecast_days))

            last_date = df_for_model['date'].max()
            f_df = pd.DataFrame({
                'date': pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days),
                'predicted_sales': preds.round().astype(int),
                'confidence_lower': (preds * 0.8).round().astype(int),
                'confidence_upper': (preds * 1.2).round().astype(int)
            })

            if scenario_active:
                st.write("Оценка ценовой эластичности и сценарный анализ...")
                elasticity = fe.estimate_price_elasticity(df_features)
                f_df['scenario_sales'] = (
                            f_df['predicted_sales'] * (1 + elasticity * price_change / 100)).round().astype(int)
                st.session_state.elasticity = elasticity

            st.session_state.forecast_results = f_df
            st.session_state.model_metrics = res
            st.session_state.forecast_done = True

            status.update(label="Прогноз успешно построен!", state="complete", expanded=False)
            st.rerun()

    # Результаты
    if st.session_state.forecast_done:
        f_df = st.session_state.forecast_results
        m = st.session_state.model_metrics

        st.header("Результаты моделирования")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Точность (sMAPE)", f"{m['mean_smape']:.1f}%")
        mc2.metric("Ошибка (RMSE)", f"{m['mean_rmse']:.1f}")
        mc3.metric("Прогнозный объем", f"{f_df['predicted_sales'].sum():,} шт")
        if 'scenario_sales' in f_df.columns:
            mc4.metric("Эластичность", f"{st.session_state.elasticity:.2f}")

        # График прогноза
        fig_res = go.Figure()
        # Доверительный интервал
        fig_res.add_trace(go.Scatter(
            x=f_df['date'].tolist() + f_df['date'].tolist()[::-1],
            y=f_df['confidence_upper'].tolist() + f_df['confidence_lower'].tolist()[::-1],
            fill='toself', fillcolor='rgba(0,176,246,0.1)', line=dict(color='rgba(255,255,255,0)'),
            name="Доверительный интервал 80%", hoverinfo='skip'
        ))
        fig_res.add_trace(
            go.Scatter(x=f_df['date'], y=f_df['predicted_sales'], name="Базовый прогноз", mode='lines+markers',
                       line=dict(color='blue')))

        if 'scenario_sales' in f_df.columns:
            fig_res.add_trace(go.Scatter(x=f_df['date'], y=f_df['scenario_sales'], name="Сценарный прогноз",
                                         line=dict(dash='dash', color='orange')))

        fig_res.update_layout(title="Прогноз спроса на заданный период", height=450)
        st.plotly_chart(fig_res, use_container_width=True)

        with st.expander("Детальные результаты кросс-валидации"):
            st.table(pd.DataFrame(m['all_folds_metrics'])[['fold', 'smape', 'rmse']])

        st.divider()
        st.subheader("Экспорт отчетов и модели")

        # Предварительная подготовка данных
        with st.spinner("Генерация файлов..."):
            csv_data = export_csv(f_df)
            pdf_data = export_pdf(f_df, m, fig=fig_res)
            xlsx_data = export_excel(f_df, m)
            docx_data = export_word(f_df, m, fig=fig_res)
            model_data = export_model(m['model'])

            # ZIP
            zip_data = create_zip_file({
                "forecast.csv": csv_data, "report.pdf": pdf_data,
                "analysis.xlsx": xlsx_data, "report.docx": docx_data
            })

        # Сетка кнопок для экспорта результатов
        col_btns = st.columns(5)
        with col_btns[0]:
            st.download_button("CSV", csv_data, "forecast.csv", use_container_width=True)
        with col_btns[1]:
            st.download_button("PDF", pdf_data, "report.pdf", use_container_width=True)
        with col_btns[2]:
            st.download_button("Excel", xlsx_data, "analysis.xlsx", use_container_width=True)
        with col_btns[3]:
            st.download_button("Word", docx_data, "report.docx", use_container_width=True)
        with col_btns[4]:
            st.download_button("ZIP", zip_data, "all_reports.zip", use_container_width=True)

        # Отдельная секция для модели
        st.info("Вы можете скачать обученную модель для использования в других Python скриптах.")
        st.download_button(
            label="Скачать обученную модель (.joblib)",
            data=model_data,
            file_name=f"model_{product_id if product_id else 'total'}.joblib",
            use_container_width=True,
            type="primary"
        )

else:
    st.info("""
    ### Инструкция по использованию
    1. Загрузите файл в формате CSV или выберите демонстрационный набор данных в боковой панели.
    2. Выберите объект анализа (конкретный товар / общий спрос).
    3. Укажите горизонт планирования (от 7 до 30 дней).
    4. При необходимости настройте сценарный анализ изменения цены.
    5. Нажмите кнопку "Построить прогноз".

    ### Требования к структуре CSV-файла
    Для корректной работы системы файл должен содержать следующие столбцы:
    - **date**: дата продажи (поддерживаются форматы YYYY-MM-DD и DD.MM.YYYY).
    - **product_id**: идентификатор товара.
    - **sales**: объем продаж (целое число).

    Опционально для сценарного анализа:
    - **price**: цена за единицу товара.

    Минимальный требуемый объем данных для обучения модели: 30 дней.
    Кодировка файла: UTF-8.
    """)

# Футер
st.markdown("---")
st.caption(
    f"Система имитационного моделирования прогнозирования спроса на товары | Random Forest | {datetime.now().year}")