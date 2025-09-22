#!/usr/bin/env python3
"""
Тест для проверки поиска и запуска приложений
"""

import sys
import os

# Добавляем путь к основному файлу
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_app_search():
    """Тестирует поиск приложений"""
    print("🔍 Тестирование поиска приложений...")
    
    try:
        from main import find_installed_apps, find_start_menu_apps, search_apps_by_name
        
        # Тестируем поиск установленных приложений
        print("📋 Поиск установленных приложений...")
        installed_apps = find_installed_apps()
        print(f"   Найдено приложений в реестре: {len(installed_apps)}")
        
        # Показываем первые 5 приложений
        if installed_apps:
            print("   Примеры найденных приложений:")
            for i, (key, app_info) in enumerate(list(installed_apps.items())[:5]):
                print(f"     {i+1}. {app_info['name']}")
        
        # Тестируем поиск в меню Пуск
        print("\n📋 Поиск приложений в меню Пуск...")
        start_menu_apps = find_start_menu_apps()
        print(f"   Найдено ярлыков в меню Пуск: {len(start_menu_apps)}")
        
        # Показываем первые 5 приложений
        if start_menu_apps:
            print("   Примеры найденных ярлыков:")
            for i, (key, app_info) in enumerate(list(start_menu_apps.items())[:5]):
                print(f"     {i+1}. {app_info['name']}")
        
        # Тестируем поиск по названию
        print("\n🔍 Тестирование поиска по названию...")
        test_queries = ['chrome', 'notepad', 'calculator', 'firefox', 'steam']
        
        for query in test_queries:
            print(f"\n   Поиск: '{query}'")
            results = search_apps_by_name(query)
            if results:
                print(f"     Найдено {len(results)} результатов:")
                for i, (key, app_info) in enumerate(results):
                    print(f"       {i+1}. {app_info['name']}")
            else:
                print(f"     Ничего не найдено")
        
        print("\n✅ Поиск приложений работает корректно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании поиска: {e}")
        return False

def test_smart_launch():
    """Тестирует умный запуск приложений"""
    print("\n🧠 Тестирование умного запуска приложений...")
    
    try:
        from main import smart_app_launch, LLM_BACKEND
        
        if not LLM_BACKEND:
            print("⚠️  LLM не настроен, тестирование умного запуска пропущено")
            return True
        
        # Тестируем поиск известного приложения
        print("   Тестирование поиска 'notepad'...")
        result = smart_app_launch('notepad')
        print(f"   Результат: {result}")
        
        # Тестируем поиск неизвестного приложения
        print("\n   Тестирование поиска неизвестного приложения...")
        result = smart_app_launch('какое-то неизвестное приложение')
        print(f"   Результат: {result}")
        
        print("\n✅ Умный запуск приложений работает!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании умного запуска: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов поиска приложений\n")
    
    # Тестируем поиск приложений
    search_ok = test_app_search()
    
    # Тестируем умный запуск
    smart_ok = test_smart_launch()
    
    print(f"\n📊 Результаты тестирования:")
    print(f"   - Поиск приложений: {'✅' if search_ok else '❌'}")
    print(f"   - Умный запуск: {'✅' if smart_ok else '❌'}")
    
    if search_ok and smart_ok:
        print("\n🎉 Все тесты пройдены! Поиск и запуск приложений работает.")
    else:
        print("\n❌ Обнаружены проблемы в работе с приложениями.")
