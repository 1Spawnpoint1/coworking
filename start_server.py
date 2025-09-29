#!/usr/bin/env python3
"""
Скрипт для запуска FastAPI сервера
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    # Проверяем, что мы в правильной директории
    if not Path("main.py").exists():
        print("Ошибка: main.py не найден. Запустите скрипт из корневой директории проекта.")
        sys.exit(1)
    
    # Создаем необходимые директории
    Path("uploads").mkdir(exist_ok=True)
    Path("cache_hyperdrive").mkdir(exist_ok=True)
    Path("production_model_assembled").mkdir(exist_ok=True)
    
    print("🚀 Запуск WB Review Moderation API...")
    print("📊 API будет доступен по адресу: http://localhost:8000")
    print("📖 Документация API: http://localhost:8000/docs")
    print("🔍 Альтернативная документация: http://localhost:8000/redoc")
    print("\n" + "="*50)
    
    # Запускаем сервер
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Автоперезагрузка при изменении кода
        log_level="info"
    )

if __name__ == "__main__":
    main()
