import io, os, zipfile
import pandas as pd
from fpdf import FPDF
from docx import Document
from docx.shared import Inches
import plotly.io as pio
from datetime import datetime
import joblib
import streamlit as st
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Настройка путей для шрифтов
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(BASE_DIR, 'fonts')


def get_font_path():
    path = os.path.join(FONTS_DIR, 'DejaVuLGCSans.ttf')
    return path if os.path.exists(path) else None


def check_fonts_and_warn():
    return get_font_path() is not None


def export_csv(df):
    """Экспорт в CSV"""
    export_df = df.copy()
    columns = {
        'date': 'Дата',
        'predicted_sales': 'Прогноз (шт)',
        'scenario_sales': 'Сценарий (шт)',
        'confidence_lower': 'Мин. граница (шт)',
        'confidence_upper': 'Макс. граница (шт)'
    }
    export_df = export_df.rename(columns={k: v for k, v in columns.items() if k in export_df.columns})
    export_df['Дата'] = export_df['Дата'].dt.date
    return export_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def export_pdf(df, metrics, fig=None):
    """Экспорт в PDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        font_path = get_font_path()

        if font_path:
            pdf.add_font('ProjectFont', '', font_path, uni=True)
            pdf.set_font('ProjectFont', '', 14)
            title = "ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ ПРОДАЖ"
        else:
            pdf.set_font('Arial', 'B', 14)
            title = "SALES FORECAST REPORT"

        pdf.cell(0, 10, title, ln=True, align='C')

        if fig:
            # График Plotly в PNG картинку
            img_bytes = pio.to_image(fig, format="png", width=1000, height=500)
            pdf.image(io.BytesIO(img_bytes), x=10, y=25, w=190)
            pdf.set_y(130)  # Отступ после графика

        pdf.set_font_size(9)
        pdf.cell(0, 7, f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True)
        pdf.cell(0, 7, f"Средняя точность (sMAPE): {metrics['mean_smape']:.2f}%", ln=True)
        pdf.ln(5)

        # Таблица
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(40, 8, "Дата", 1, 0, 'C', fill=True)
        pdf.cell(50, 8, "Прогноз (шт)", 1, 0, 'C', fill=True)
        col_name = "Сценарий (шт)" if 'scenario_sales' in df.columns else "Верх. граница"
        pdf.cell(50, 8, col_name, 1, 1, 'C', fill=True)

        pdf.set_font_size(8)
        for _, row in df.iterrows():
            pdf.cell(40, 7, str(row['date'].date()), 1, 0, 'C')
            pdf.cell(50, 7, str(int(row['predicted_sales'])), 1, 0, 'C')
            val = row['scenario_sales'] if 'scenario_sales' in df.columns else row['confidence_upper']
            pdf.cell(50, 7, str(int(val)), 1, 1, 'C')

        return bytes(pdf.output())
    except Exception as e:
        st.error(f"Ошибка PDF: {e}")
        return None


def export_word(df, metrics, fig=None):
    """Экспорт в Word"""
    try:
        doc = Document()

        title = doc.add_heading('ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ ПРОДАЖ', level=0)

        title_run = title.runs[0]
        title_run.font.size = Pt(20)
        title_run.font.bold = True

        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        if fig:
            img_bytes = pio.to_image(fig, format="png", width=1000, height=500)
            doc.add_picture(io.BytesIO(img_bytes), width=Inches(6))

        doc.add_paragraph(f"Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.add_paragraph(f"Средняя точность модели (sMAPE): {metrics['mean_smape']:.2f}%")

        table = doc.add_table(rows=1, cols=3)
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Дата'
        hdr_cells[1].text = 'Прогноз (шт)'
        hdr_cells[2].text = 'Сценарий/Граница'

        for _, row in df.iterrows():
            row_cells = table.add_row().cells
            row_cells[0].text = str(row['date'].date())
            row_cells[1].text = str(int(row['predicted_sales']))
            val = row['scenario_sales'] if 'scenario_sales' in df.columns else row['confidence_upper']
            row_cells[2].text = str(int(val))

        target = io.BytesIO()
        doc.save(target)
        return target.getvalue()
    except Exception as e:
        st.error(f"Ошибка Word: {e}")
        return None


def export_excel(df, metrics):
    """Экспорт в Excel"""
    output = io.BytesIO()

    # Подготовка данных для первого листа (Детальный прогноз)
    df_forecast = df.copy()
    columns_map = {
        'date': 'Дата',
        'predicted_sales': 'Прогноз (шт)',
        'scenario_sales': 'Сценарий (шт)',
        'confidence_lower': 'Нижняя граница (шт)',
        'confidence_upper': 'Верхняя граница (шт)'
    }
    df_forecast = df_forecast.rename(columns={k: v for k, v in columns_map.items() if k in df_forecast.columns})
    df_forecast['Дата'] = df_forecast['Дата'].dt.date

    # Подготовка данных для второго листа (Метрики модели)
    metrics_data = [
        ['Наименование показателя', 'Значение'],
        ['Заголовок отчета', 'ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ ПРОДАЖ'],
        ['Алгоритм', 'Random Forest Regressor'],
        ['Средняя точность (sMAPE)', f"{metrics['mean_smape']:.2f}%"],
        ['Среднеквадратичная ошибка (RMSE)', round(metrics['mean_rmse'], 2)],
        ['Количество фолдов кросс-валидации', len(metrics['all_folds_metrics'])],
        ['Дата формирования отчета', datetime.now().strftime('%d.%m.%Y %H:%M')]
    ]
    df_metrics = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])

    # Подготовка данных для третьего листа (Аналитика)
    stats_data = [
        ['Показатель спроса', 'Значение'],
        ['Общий прогнозный объем (ед.)', df['predicted_sales'].sum()],
        ['Среднесуточные продажи (ед.)', round(df['predicted_sales'].mean(), 2)],
        ['Максимальный суточный спрос (ед.)', df['predicted_sales'].max()],
        ['Минимальный суточный спрос (ед.)', df['predicted_sales'].min()],
        ['Вариативность прогноза (%)', f"{round(df['predicted_sales'].std() / df['predicted_sales'].mean() * 100, 1)}%"]
    ]
    df_stats = pd.DataFrame(stats_data[1:], columns=stats_data[0])

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_forecast.to_excel(writer, sheet_name='Детальный прогноз', index=False)
        df_metrics.to_excel(writer, sheet_name='Метрики модели', index=False)
        df_stats.to_excel(writer, sheet_name='Бизнес-аналитика', index=False)

        # Добавление стилей через openpyxl
        workbook = writer.book
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]

            # Настройка ширины колонок
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                worksheet.column_dimensions[column].width = max_length + 5

            # Оформление заголовков
            from openpyxl.styles import Font, PatternFill, Alignment
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")

            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

            # Закрепление верхней строки
            worksheet.freeze_panes = 'A2'

    return output.getvalue()


def create_zip_file(files):
    """Создание ZIP архива"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if data:
                z.writestr(name, data)
    return buf.getvalue()


def export_model(model):
    """Экспорт обученной модели .joblib"""
    try:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        return buffer.getvalue()
    except:
        return None