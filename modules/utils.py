# ==========================================================
# Файл: utils.py
# Описание: Вспомогательный модуль с общими функциями для установки
#           зависимостей и других служебных операций.
# Соответствует:
#   - Диаграмме компонентов (Component Diagram) как сервисный модуль
#   - Нефункциональному требованию НФТ3.3 (устойчивость к отсутствию интернета)
# Автор: [Ваше имя]
# Дата: 2025-12-12
# Версия: 1.0
# ==========================================================

import subprocess
import sys
import importlib

def install_package(package_name):
    """Устанавливает пакет если он не установлен"""
    try:
        importlib.import_module(package_name.replace('-', '_'))
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except:
            return False