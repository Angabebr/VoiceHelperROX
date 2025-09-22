#!/usr/bin/env python3
"""
Простой тест для проверки интеграции LLM в голосовой помощник
"""

import sys
import os

# Добавляем путь к основному файлу
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_llm_integration():
    """Тестирует интеграцию с LLM"""
    print("🔍 Тестирование интеграции LLM...")
    
    try:
        # Импортируем функции из основного файла
        from main import check_llm_status, ask_llm, LLM_BACKEND
        
        print(f"📊 Текущий LLM бэкенд: {LLM_BACKEND or 'Не настроен'}")
        
        # Проверяем статус
        status = check_llm_status()
        print(f"📋 Статус LLM:")
        print(f"   - Ollama: {'✅' if status['ollama'] else '❌'}")
        print(f"   - Transformers: {'✅' if status['transformers'] else '❌'}")
        print(f"   - Активный бэкенд: {status['active_backend'] or 'Нет'}")
        
        # Тестируем простой вопрос
        if LLM_BACKEND:
            print("\n🤖 Тестирование простого вопроса...")
            test_question = "Что такое Python?"
            print(f"Вопрос: {test_question}")
            
            answer = ask_llm(test_question)
            print(f"Ответ: {answer}")
            
            if "недоступна" in answer.lower() or "ошибка" in answer.lower():
                print("⚠️  LLM не работает корректно")
                return False
            else:
                print("✅ LLM работает корректно")
                return True
        else:
            print("⚠️  LLM не настроен")
            return False
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def test_basic_functions():
    """Тестирует базовые функции без LLM"""
    print("\n🔧 Тестирование базовых функций...")
    
    try:
        from main import safe_eval_math, speak
        
        # Тест математических вычислений
        math_result = safe_eval_math("2 + 2")
        print(f"Математика (2+2): {math_result}")
        
        # Тест синтеза речи (без вывода звука)
        print("Тест синтеза речи: 'Тест'")
        speak("Тест")
        
        print("✅ Базовые функции работают")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка базовых функций: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов голосового помощника\n")
    
    # Тестируем базовые функции
    basic_ok = test_basic_functions()
    
    # Тестируем LLM
    llm_ok = test_llm_integration()
    
    print(f"\n📊 Результаты тестирования:")
    print(f"   - Базовые функции: {'✅' if basic_ok else '❌'}")
    print(f"   - LLM интеграция: {'✅' if llm_ok else '❌'}")
    
    if basic_ok and llm_ok:
        print("\n🎉 Все тесты пройдены! Голосовой помощник готов к работе.")
    elif basic_ok:
        print("\n⚠️  Базовые функции работают, но LLM не настроен.")
        print("   Для расширенных возможностей установите Ollama или transformers.")
    else:
        print("\n❌ Обнаружены проблемы. Проверьте установку зависимостей.")

