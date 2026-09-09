@echo off
echo Запуск системы прогнозирования спроса

if not exist venv (
    echo Создание виртуального окружения...
    python -m venv venv
)

echo Активация окружения и проверка библиотек...
call venv\Scripts\activate
pip install -r requirements.txt

echo Запуск приложения Streamlit...
streamlit run app.py

pause