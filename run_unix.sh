#!/bin/bash
echo "Запуск системы прогнозирования спроса"

if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Установка зависимостей..."
pip install -r requirements.txt

echo "Запуск приложения..."
streamlit run app.py