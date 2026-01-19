# ==========================================================
# Файл: export.py
# Описание: Модуль экспорта результатов прогнозирования.
#           Обеспечивает генерацию отчетов в форматах CSV, PDF, Excel, Word,
#           создание ZIP-архивов и проверку доступности шрифтов для PDF.
# Соответствует:
#   - Диаграмме компонентов (Component Diagram)
#   - Функциональным требованиям ФТ4.1-ФТ4.4
#   - Бизнес-требованию БТ2.3 (интеграция отчетов в бизнес-процессы)
#   - Интерфейсным требованиям ИТ4.1 (индикаторы), ИТ2.1-ИТ2.2 (кнопки)
#   - Нефункциональным требованиям НФТ3.2 (обработка ошибок), НФТ5.1 (русский язык)
# Автор: Бешляга Дарья Денисовна
# Дата: 2025-12-12
# Версия: 2.0
# ==========================================================

import io
import zipfile
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from modules.utils import install_package

# Определение абсолютного пути к папке со шрифтами
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(BASE_DIR, 'fonts')


def get_project_font_path():
    """Получает путь к шрифту из папки проекта"""
    # Проверяет наличие шрифтов в папке fonts проекта
    font_paths = [
        os.path.join(FONTS_DIR, 'DejaVuLGCSans.ttf'),
        os.path.join(FONTS_DIR, 'DejaVuLGCSans-Bold.ttf'),
        os.path.join(FONTS_DIR, 'DejaVuLGCSans-Oblique.ttf'),
    ]

    # Проверяет наличие файлов
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path

    # Если шрифты не найдены в папке проекта, возвращает None
    return None


def check_fonts_available():
    """Проверяет доступность шрифтов"""
    font_path = get_project_font_path()

    if font_path:
        font_name = os.path.basename(font_path)
        return True, f"Используется шрифт: {font_name}"
    else:
        # Проверет, существует ли папка fonts
        if os.path.exists(FONTS_DIR):
            return False, "Папка fonts существует, но файлы шрифтов не найдены. Убедитесь, что в папке fonts есть файлы шрифтов (DejaVuLGCSans.ttf)."
        else:
            return False, "Папка fonts не найдена. Убедитесь, что в проекте есть папка fonts с файлами шрифтов."


def export_csv(forecast_df):
    """Экспорт в CSV с русскими названиями столбцов"""
    try:
        csv_df = forecast_df.copy()

        if 'scenario_sales' in csv_df.columns:
            csv_df = csv_df[['date', 'predicted_sales', 'scenario_sales']].copy()
            csv_df.columns = ['Дата', 'Базовый прогноз (шт)', 'Сценарный прогноз (шт)']
        else:
            csv_df = csv_df[['date', 'predicted_sales', 'confidence_lower', 'confidence_upper']].copy()
            csv_df.columns = ['Дата', 'Прогноз продаж (шт)', 'Нижняя граница', 'Верхняя граница']

        csv_df['Дата'] = csv_df['Дата'].dt.strftime('%d.%m.%Y')
        csv = csv_df.to_csv(index=False, encoding='utf-8-sig')
        return csv.encode('utf-8-sig')
    except Exception as e:
        st.error(f"Ошибка при создании CSV: {e}")
        return None


def export_pdf(forecast_df, metrics):
    """Экспорт в PDF"""
    try:
        # Проверка и установка зависимостей
        if not install_package('fpdf2'):
            st.warning("Для создания PDF необходимо установить fpdf2")
            return None

        from fpdf import FPDF

        # Поиск шрифта в папке проекта
        font_path = get_project_font_path()
        font_available = font_path is not None

        # Класс PDF с обработкой шрифтов
        class PDF(FPDF):
            def __init__(self, use_project_font=False, font_path=None):
                super().__init__()
                self.use_project_font = use_project_font
                self.font_path = font_path

                if use_project_font and font_path:
                    try:
                        # Регистрирует шрифт из проекта
                        self.add_font('ProjectFont', '', font_path, uni=True)
                        self.add_font('ProjectFont', 'B', font_path, uni=True)
                        self.add_font('ProjectFont', 'I', font_path, uni=True)
                    except Exception as e:
                        st.warning(f"Не удалось загрузить шрифт из проекта: {e}")
                        self.use_project_font = False

            def header(self):
                """Заголовок отчета"""
                if self.use_project_font:
                    try:
                        self.set_font('ProjectFont', 'B', 16)
                        self.cell(0, 10, 'ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ ПРОДАЖ', 0, 1, 'C')
                    except:
                        self.set_font('Arial', 'B', 16)
                        self.cell(0, 10, 'SALES FORECAST REPORT', 0, 1, 'C')
                else:
                    self.set_font('Arial', 'B', 16)
                    self.cell(0, 10, 'SALES FORECAST REPORT', 0, 1, 'C')
                self.ln(10)

            def footer(self):
                """Футер с нумерацией страниц"""
                self.set_y(-15)
                if self.use_project_font:
                    try:
                        self.set_font('ProjectFont', '', 8)
                        self.cell(0, 10, f'Страница {self.page_no()}', 0, 0, 'C')
                    except:
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
                else:
                    self.set_font('Arial', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            def safe_cell(self, w, h, txt, border=0, ln=0, align='L'):
                """Создание ячейки с обработкой кодировки"""
                try:
                    if self.use_project_font:
                        self.cell(w, h, txt, border, ln, align)
                    else:
                        # Для английской версии транслитерирует русский текст
                        if 'ОТЧЕТ' in txt or 'ПРОГНОЗ' in txt or 'Страница' in txt:
                            self.cell(w, h, txt, border, ln, align)
                        else:
                            self.cell(w, h, txt, border, ln, align)
                except Exception:
                    # Если ошибка, использует латинские символы
                    safe_txt = txt.encode('latin-1', 'replace').decode('latin-1')
                    self.cell(w, h, safe_txt, border, ln, align)

        # Создает PDF
        pdf = PDF(use_project_font=font_available, font_path=font_path)
        pdf.add_page()

        # Определяет функцию для добавления текста
        def add_text(text, style=('', 12)):
            font_style, font_size = style

            if pdf.use_project_font:
                try:
                    pdf.set_font('ProjectFont', font_style, font_size)
                except:
                    pdf.set_font('Arial', font_style, font_size)
            else:
                pdf.set_font('Arial', font_style, font_size)

            pdf.cell(0, 10, text, 0, 1)

        # Заголовочная информация
        if pdf.use_project_font:
            add_text(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
            add_text(f"Период прогноза: {len(forecast_df)} дней")
            analysis_type = st.session_state.get('analysis_type', 'Один товар')
            add_text(f"Тип анализа: {analysis_type}")

            # Информация о сценарии
            if 'scenario_params' in st.session_state:
                params = st.session_state.scenario_params
                add_text(f"Сценарный анализ: изменение цены на {params['price_change']:+.1f}%")
                add_text(f"Оцененная эластичность: {params['elasticity']:.2f}")
        else:
            # Английская версия
            add_text(f"Report date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            add_text(f"Forecast period: {len(forecast_df)} days")
            analysis_type = st.session_state.get('analysis_type', 'Один товар')
            if analysis_type == 'Один товар':
                add_text("Analysis type: Single product")
            else:
                add_text("Analysis type: All products")

            # Информация о сценарии
            if 'scenario_params' in st.session_state:
                params = st.session_state.scenario_params
                add_text(f"Scenario analysis: price change {params['price_change']:+.1f}%")
                add_text(f"Estimated elasticity: {params['elasticity']:.2f}")

        pdf.ln(5)

        # Таблица прогноза
        if pdf.use_project_font:
            add_text('Прогноз продаж по дням:', ('B', 14))
        else:
            add_text('Daily Sales Forecast:', ('B', 14))

        # Установка ширины колонок
        col_width = 40
        row_height = 8

        # Заголовки таблицы
        if pdf.use_project_font:
            try:
                pdf.set_font('ProjectFont', 'B', 10)
            except:
                pdf.set_font('Arial', 'B', 10)
        else:
            pdf.set_font('Arial', 'B', 10)

        if 'scenario_sales' in forecast_df.columns:
            if pdf.use_project_font:
                pdf.cell(col_width, row_height, 'Дата', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Базовый', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Сценарий', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Разница', 1, 1, 'C')
            else:
                pdf.cell(col_width, row_height, 'Date', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Base', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Scenario', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Difference', 1, 1, 'C')

            # Данные таблицы
            if pdf.use_project_font:
                try:
                    pdf.set_font('ProjectFont', '', 9)
                except:
                    pdf.set_font('Arial', '', 9)
            else:
                pdf.set_font('Arial', '', 9)

            for _, row in forecast_df.iterrows():
                date_str = row['date'].strftime('%d.%m.%Y')
                diff = row['scenario_sales'] - row['predicted_sales']
                diff_pct = round(diff / row['predicted_sales'] * 100, 2) if row['predicted_sales'] > 0 else 0

                pdf.cell(col_width, row_height, date_str, 1, 0, 'C')
                pdf.cell(col_width, row_height, f"{row['predicted_sales']:,}", 1, 0, 'R')
                pdf.cell(col_width, row_height, f"{row['scenario_sales']:,}", 1, 0, 'R')
                if pdf.use_project_font:
                    pdf.cell(col_width, row_height, f"{diff:+,} ({diff_pct:+.1f}%)", 1, 1, 'R')
                else:
                    pdf.cell(col_width, row_height, f"{diff:+,} ({diff_pct:+.1f}%)", 1, 1, 'R')
        else:
            if pdf.use_project_font:
                pdf.cell(col_width, row_height, 'Дата', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Прогноз', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Мин', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Макс', 1, 1, 'C')
            else:
                pdf.cell(col_width, row_height, 'Date', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Forecast', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Min', 1, 0, 'C')
                pdf.cell(col_width, row_height, 'Max', 1, 1, 'C')

            # Данные таблицы
            if pdf.use_project_font:
                try:
                    pdf.set_font('ProjectFont', '', 9)
                except:
                    pdf.set_font('Arial', '', 9)
            else:
                pdf.set_font('Arial', '', 9)

            for _, row in forecast_df.iterrows():
                date_str = row['date'].strftime('%d.%m.%Y')
                pdf.cell(col_width, row_height, date_str, 1, 0, 'C')
                pdf.cell(col_width, row_height, f"{row['predicted_sales']:,}", 1, 0, 'R')
                pdf.cell(col_width, row_height, f"{row['confidence_lower']:,}", 1, 0, 'R')
                pdf.cell(col_width, row_height, f"{row['confidence_upper']:,}", 1, 1, 'R')

        pdf.ln(10)

        # Метрики точности модели
        if metrics:
            if pdf.use_project_font:
                add_text('Метрики точности модели:', ('B', 14))
                add_text(f"sMAPE: {metrics.get('smape', 0):.1f}%")
                add_text(f"RMSE: {metrics.get('rmse', 0):.1f}")
                add_text(f"Средний sMAPE: {metrics.get('mean_smape', 0):.1f}%")
                add_text(f"Средний RMSE: {metrics.get('mean_rmse', 0):.1f}")
            else:
                add_text('Model Accuracy Metrics:', ('B', 14))
                add_text(f"sMAPE: {metrics.get('smape', 0):.1f}%")
                add_text(f"RMSE: {metrics.get('rmse', 0):.1f}")
                add_text(f"Mean sMAPE: {metrics.get('mean_smape', 0):.1f}%")
                add_text(f"Mean RMSE: {metrics.get('mean_rmse', 0):.1f}")
            pdf.ln(5)

        # Сводка прогноза
        if pdf.use_project_font:
            add_text('Сводка прогноза:', ('B', 14))
        else:
            add_text('Forecast Summary:', ('B', 14))

        if 'scenario_sales' in forecast_df.columns:
            base_mean = forecast_df['predicted_sales'].mean()
            scenario_mean = forecast_df['scenario_sales'].mean()
            base_total = forecast_df['predicted_sales'].sum()
            scenario_total = forecast_df['scenario_sales'].sum()

            if pdf.use_project_font:
                add_text(f"Средние продажи в день (базовый): {base_mean:.0f} шт")
                add_text(f"Средние продажи в день (сценарий): {scenario_mean:.0f} шт")
                add_text(f"Изменение: {((scenario_mean / base_mean - 1) * 100):+.1f}%")
                add_text(f"Всего за период (базовый): {base_total:.0f} шт")
                add_text(f"Всего за период (сценарий): {scenario_total:.0f} шт")
                add_text(f"Изменение: {((scenario_total / base_total - 1) * 100):+.1f}%")
            else:
                add_text(f"Average daily sales (base): {base_mean:.0f} units")
                add_text(f"Average daily sales (scenario): {scenario_mean:.0f} units")
                add_text(f"Change: {((scenario_mean / base_mean - 1) * 100):+.1f}%")
                add_text(f"Total for period (base): {base_total:.0f} units")
                add_text(f"Total for period (scenario): {scenario_total:.0f} units")
                add_text(f"Change: {((scenario_total / base_total - 1) * 100):+.1f}%")
        else:
            if pdf.use_project_font:
                add_text(f"Средние продажи в день: {forecast_df['predicted_sales'].mean():.0f} шт")
                add_text(f"Всего за период: {forecast_df['predicted_sales'].sum():.0f} шт")
                add_text(f"Максимум за день: {forecast_df['predicted_sales'].max():.0f} шт")
                add_text(f"Минимум за день: {forecast_df['predicted_sales'].min():.0f} шт")
                add_text(f"Дней прогноза: {len(forecast_df)}")
            else:
                add_text(f"Average daily sales: {forecast_df['predicted_sales'].mean():.0f} units")
                add_text(f"Total for period: {forecast_df['predicted_sales'].sum():.0f} units")
                add_text(f"Maximum per day: {forecast_df['predicted_sales'].max():.0f} units")
                add_text(f"Minimum per day: {forecast_df['predicted_sales'].min():.0f} units")
                add_text(f"Forecast days: {len(forecast_df)}")

        # Сохранение в буфер
        pdf_buffer = io.BytesIO()
        pdf_output = pdf.output()

        if isinstance(pdf_output, str):
            pdf_buffer.write(pdf_output.encode('latin-1'))
        else:
            pdf_buffer.write(pdf_output)

        pdf_buffer.seek(0)

        # Выводит информацию о шрифтах
        if not font_available:
            st.info(
                "PDF создан с использованием стандартных шрифтов Arial (английская версия). Для русского текста добавьте шрифты в папку fonts.")

        return pdf_buffer.getvalue()

    except Exception as e:
        st.error(f"Ошибка при создании PDF: {e}")
        return None


def export_word(forecast_df, metrics):
    """Экспорт в Word документ"""
    try:
        # Проверка зависимостей
        if not install_package('python-docx'):
            st.warning("Для создания Word документа необходимо установить python-docx")
            return None

        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT

        doc = Document()

        # Настройка стилей
        style = doc.styles['Normal']
        style.font.name = 'Arial'
        style.font.size = Pt(10)

        # Заголовок
        title = doc.add_heading('Отчет по прогнозированию продаж', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title.runs[0]
        title_run.font.name = 'Arial'
        title_run.font.size = Pt(16)
        title_run.font.bold = True

        # Информация об отчете
        doc.add_paragraph(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.add_paragraph(f"Период прогноза: {len(forecast_df)} дней")
        analysis_type = st.session_state.get('analysis_type', 'Один товар')
        doc.add_paragraph(f"Тип анализа: {analysis_type}")

        # Информация о сценарии
        if 'scenario_params' in st.session_state:
            params = st.session_state.scenario_params
            doc.add_paragraph(f"Сценарный анализ: изменение цены на {params['price_change']:+.1f}%")
            doc.add_paragraph(f"Оцененная эластичность: {params['elasticity']:.2f}")

        doc.add_paragraph()

        # Метрики модели
        if metrics:
            doc.add_heading('Метрики точности модели', level=1)
            metrics_table = doc.add_table(rows=5, cols=2)
            metrics_table.alignment = WD_TABLE_ALIGNMENT.CENTER

            # Заголовки
            hdr_cells = metrics_table.rows[0].cells
            hdr_cells[0].text = 'Метрика'
            hdr_cells[1].text = 'Значение'

            # Жирный шрифт для заголовков
            for cell in hdr_cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True

            # Данные
            metrics_data = [
                ('sMAPE', f"{metrics.get('smape', 0):.1f}%"),
                ('RMSE', f"{metrics.get('rmse', 0):.1f}"),
                ('Средний sMAPE', f"{metrics.get('mean_smape', 0):.1f}%"),
                ('Средний RMSE', f"{metrics.get('mean_rmse', 0):.1f}")
            ]

            for i, (metric, value) in enumerate(metrics_data, 1):
                row_cells = metrics_table.rows[i].cells
                row_cells[0].text = metric
                row_cells[1].text = value

        doc.add_paragraph()

        # Детальный прогноз
        doc.add_heading('Детальный прогноз по дням', level=1)

        if 'scenario_sales' in forecast_df.columns:
            forecast_table = doc.add_table(rows=min(len(forecast_df) + 1, 31), cols=5)

            headers = ['Дата', 'Базовый прогноз (шт)', 'Сценарный прогноз (шт)', 'Разница (шт)', 'Разница (%)']

            # Заголовки
            hdr_cells = forecast_table.rows[0].cells
            for i, header in enumerate(headers):
                hdr_cells[i].text = header
                for paragraph in hdr_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True

            # Данные
            max_rows = min(30, len(forecast_df))
            for i in range(max_rows):
                row = forecast_df.iloc[i]
                diff = row['scenario_sales'] - row['predicted_sales']
                diff_pct = round(diff / row['predicted_sales'] * 100, 2) if row['predicted_sales'] > 0 else 0

                row_cells = forecast_table.rows[i + 1].cells
                row_cells[0].text = row['date'].strftime('%d.%m.%Y')
                row_cells[1].text = f"{row['predicted_sales']:,}"
                row_cells[2].text = f"{row['scenario_sales']:,}"
                row_cells[3].text = f"{diff:+,}"
                row_cells[4].text = f"{diff_pct:+.2f}%"
        else:
            forecast_table = doc.add_table(rows=min(len(forecast_df) + 1, 31), cols=4)

            headers = ['Дата', 'Прогноз (шт)', 'Нижняя граница', 'Верхняя граница']

            # Заголовки
            hdr_cells = forecast_table.rows[0].cells
            for i, header in enumerate(headers):
                hdr_cells[i].text = header
                for paragraph in hdr_cells[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True

            # Данные
            max_rows = min(30, len(forecast_df))
            for i in range(max_rows):
                row = forecast_df.iloc[i]
                row_cells = forecast_table.rows[i + 1].cells
                row_cells[0].text = row['date'].strftime('%d.%m.%Y')
                row_cells[1].text = f"{row['predicted_sales']:,}"
                row_cells[2].text = f"{row['confidence_lower']:,}"
                row_cells[3].text = f"{row['confidence_upper']:,}"

        doc.add_paragraph()

        # Сводные метрики
        doc.add_heading('Сводные метрики прогноза', level=1)

        if 'scenario_sales' in forecast_df.columns:
            summary_table = doc.add_table(rows=7, cols=2)

            base_mean = forecast_df['predicted_sales'].mean()
            scenario_mean = forecast_df['scenario_sales'].mean()
            base_total = forecast_df['predicted_sales'].sum()
            scenario_total = forecast_df['scenario_sales'].sum()

            summary_data = [
                ('Показатель', 'Значение'),
                ('Средние продажи в день (базовый)', f"{base_mean:.0f} шт"),
                ('Средние продажи в день (сценарий)', f"{scenario_mean:.0f} шт"),
                ('Изменение средних продаж', f"{((scenario_mean / base_mean - 1) * 100):+.1f}%"),
                ('Всего за период (базовый)', f"{base_total:.0f} шт"),
                ('Всего за период (сценарий)', f"{scenario_total:.0f} шт"),
                ('Изменение общего объема', f"{((scenario_total / base_total - 1) * 100):+.1f}%")
            ]
        else:
            summary_table = doc.add_table(rows=6, cols=2)

            summary_data = [
                ('Показатель', 'Значение'),
                ('Средние продажи в день', f"{forecast_df['predicted_sales'].mean():.0f} шт"),
                ('Всего за период', f"{forecast_df['predicted_sales'].sum():.0f} шт"),
                ('Максимум за день', f"{forecast_df['predicted_sales'].max():.0f} шт"),
                ('Минимум за день', f"{forecast_df['predicted_sales'].min():.0f} шт"),
                ('Дней прогноза', f"{len(forecast_df)}")
            ]

        for i, (param, value) in enumerate(summary_data):
            row_cells = summary_table.rows[i].cells
            row_cells[0].text = param
            row_cells[1].text = value

            # Жирный шрифт для первой строки
            if i == 0:
                for cell in row_cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.bold = True

        # Сохранение в буфер
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        doc_buffer.seek(0)

        return doc_buffer.getvalue()

    except Exception as e:
        st.error(f"Ошибка при создании Word документа: {e}")
        return None


def export_excel(forecast_df, metrics):
    """Экспорт в Excel"""
    try:
        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Основные данные прогноза
            forecast_display = forecast_df.copy()
            forecast_display['date'] = forecast_display['date'].dt.strftime('%d.%m.%Y')

            if 'scenario_sales' in forecast_display.columns:
                forecast_display['Разница (шт)'] = forecast_display['scenario_sales'] - forecast_display[
                    'predicted_sales']
                forecast_display['Разница (%)'] = (
                            forecast_display['Разница (шт)'] / forecast_display['predicted_sales'] * 100).round(1)

                forecast_display = forecast_display[
                    ['date', 'predicted_sales', 'scenario_sales', 'Разница (шт)', 'Разница (%)']
                ]
                forecast_display.columns = ['Дата', 'Базовый прогноз (шт)', 'Сценарный прогноз (шт)', 'Разница (шт)',
                                            'Разница (%)']
            else:
                forecast_display = forecast_display[['date', 'predicted_sales', 'confidence_lower', 'confidence_upper']]
                forecast_display.columns = ['Дата', 'Прогноз (шт)', 'Нижняя граница', 'Верхняя граница']

            forecast_display.to_excel(writer, sheet_name='Прогноз', index=False)

            # Метрики точности
            if metrics:
                metrics_df = pd.DataFrame({
                    'Метрика': ['sMAPE', 'RMSE', 'Средний sMAPE', 'Средний RMSE'],
                    'Значение': [
                        f"{metrics.get('smape', 0):.1f}%",
                        f"{metrics.get('rmse', 0):.1f}",
                        f"{metrics.get('mean_smape', 0):.1f}%",
                        f"{metrics.get('mean_rmse', 0):.1f}"
                    ]
                })
                metrics_df.to_excel(writer, sheet_name='Метрики', index=False)

            # Сводная статистика
            if 'scenario_sales' in forecast_df.columns:
                base_mean = forecast_df['predicted_sales'].mean()
                scenario_mean = forecast_df['scenario_sales'].mean()
                base_total = forecast_df['predicted_sales'].sum()
                scenario_total = forecast_df['scenario_sales'].sum()

                summary_df = pd.DataFrame({
                    'Показатель': [
                        'Средние продажи в день (базовый)',
                        'Средние продажи в день (сценарий)',
                        'Изменение средних продаж',
                        'Всего за период (базовый)',
                        'Всего за период (сценарий)',
                        'Изменение общего объема',
                        'Дней прогноза'
                    ],
                    'Значение': [
                        f"{base_mean:.0f} шт",
                        f"{scenario_mean:.0f} шт",
                        f"{((scenario_mean / base_mean - 1) * 100):+.1f}%",
                        f"{base_total:.0f} шт",
                        f"{scenario_total:.0f} шт",
                        f"{((scenario_total / base_total - 1) * 100):+.1f}%",
                        f"{len(forecast_df)}"
                    ]
                })
            else:
                summary_df = pd.DataFrame({
                    'Показатель': [
                        'Средние продажи в день',
                        'Всего за период',
                        'Максимум за день',
                        'Минимум за день',
                        'Дней прогноза'
                    ],
                    'Значение': [
                        f"{forecast_df['predicted_sales'].mean():.0f} шт",
                        f"{forecast_df['predicted_sales'].sum():.0f} шт",
                        f"{forecast_df['predicted_sales'].max():.0f} шт",
                        f"{forecast_df['predicted_sales'].min():.0f} шт",
                        f"{len(forecast_df)}"
                    ]
                })

            summary_df.to_excel(writer, sheet_name='Статистика', index=False)

        excel_buffer.seek(0)
        return excel_buffer.getvalue()

    except Exception as e:
        st.error(f"Ошибка при создании Excel: {e}")
        return None


def create_zip_file(files_dict):
    """Создает ZIP архив с файлами"""
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
            for filename, file_data in files_dict.items():
                if file_data:
                    zip_file.writestr(filename, file_data)
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    except Exception as e:
        st.error(f"Ошибка при создании ZIP архива: {e}")
        return None


# Дополнительные функции для проверки шрифтов
def check_fonts_and_warn():
    """Проверяет шрифты и выводит предупреждение если их нет"""
    font_available, message = check_fonts_available()

    if not font_available:
        st.warning(f"{message}")
        st.info("""
        **Рекомендация:** 
        1. Создайте папку `fonts` в корне проекта
        2. Добавьте файлы шрифтов (например, DejaVuLGCSans.ttf)
        3. Перезапустите приложение

        Без шрифтов PDF будет создаваться на английском языке.
        """)

    return font_available


EXPORT_FUNCTIONS = {
    'csv': export_csv,
    'pdf': export_pdf,
    'word': export_word,
    'excel': export_excel
}