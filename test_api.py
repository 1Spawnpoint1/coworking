#!/usr/bin/env python3
"""
Простой тест для проверки работы API
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Тестирование WB Review Moderation API")
    print("=" * 50)
    
    # Тест 1: Проверка статуса
    print("1. Проверка статуса API...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ API работает")
            print(f"   Ответ: {response.json()}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return
    
    # Тест 2: Проверка здоровья
    print("\n2. Проверка здоровья...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Система здорова")
            print(f"   Модель загружена: {health_data.get('model_loaded', False)}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 3: Анализ текста
    print("\n3. Тестирование анализа текста...")
    test_texts = [
        "Отличный товар! Рекомендую всем!",
        "Это полная хуйня, не покупайте!",
        "Качество нормальное, но доставка медленная."
    ]
    
    for text in test_texts:
        try:
            response = requests.post(f"{base_url}/analyze", params={"text": text})
            if response.status_code == 200:
                result = response.json()
                verdict = "ТОКСИЧНО" if result['label'] == 1 else "Нормально"
                print(f"   '{text[:30]}...' -> {verdict} ({result['probability']:.2%})")
            else:
                print(f"   ❌ Ошибка анализа: {response.status_code}")
                if response.status_code == 422:
                    print(f"   Детали: {response.text}")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
    
    # Тест 4: Статистика дашборда
    print("\n4. Получение статистики дашборда...")
    try:
        response = requests.get(f"{base_url}/dashboard/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Статистика получена")
            print(f"   Всего отзывов: {stats['total_reviews']}")
            print(f"   Точность: {stats['accuracy']}%")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 4.1: Статистика обработанных файлов
    print("\n4.1. Получение статистики обработанных файлов...")
    try:
        response = requests.get(f"{base_url}/dashboard/processed-files")
        if response.status_code == 200:
            files_stats = response.json()
            print("✅ Статистика файлов получена")
            print(f"   Обработано файлов: {files_stats['total_files']}")
            print(f"   Всего отзывов: {files_stats['total_reviews']}")
            print(f"   Токсичных: {files_stats['flagged_reviews']}")
            print(f"   Чистых: {files_stats['clean_reviews']}")
            print(f"   Процент токсичности: {files_stats['flag_rate']}%")
            print(f"   Время обработки: {files_stats['total_processing_time']}с")
            if files_stats['total_files'] > 0:
                print(f"   Последние файлы:")
                for file_info in files_stats['files'][-3:]:  # Показываем последние 3 файла
                    print(f"     - {file_info['filename']}: {file_info['total_reviews']} отзывов")
            else:
                print(f"   {files_stats.get('message', 'Нет обработанных файлов')}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тест 5: Данные для графиков
    print("\n5. Получение данных для графиков...")
    try:
        response = requests.get(f"{base_url}/dashboard/charts")
        if response.status_code == 200:
            charts = response.json()
            print("✅ Данные графиков получены")
            print(f"   Классификация: {charts['classification']['data']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_api()
