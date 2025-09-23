#!/usr/bin/env python3
"""
Тест микрофона и настроек распознавания речи
"""

import sys
import os

# Добавляем путь к основному файлу
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_microphone_settings():
    """Тестирует настройки микрофона"""
    print("🎤 Тестирование микрофона и настроек распознавания речи\n")
    
    try:
        from main import setup_microphone, listen_speech_recognition
        
        # Настройка микрофона
        print("1. Настройка микрофона...")
        mic_ok = setup_microphone()
        
        if not mic_ok:
            print("❌ Не удалось настроить микрофон")
            return False
        
        # Тест распознавания речи
        print("\n2. Тест распознавания речи...")
        print("📢 Скажите что-нибудь (например: 'привет' или 'тест')")
        print("⏱️ У вас есть 15 секунд...")
        
        result = listen_speech_recognition(timeout=15)
        
        if result:
            print(f"✅ Распознано: '{result}'")
            return True
        else:
            print("❌ Ничего не распознано")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

def test_speech_recognition():
    """Тестирует различные настройки распознавания"""
    print("\n🔧 Тестирование настроек распознавания речи...")
    
    try:
        import speech_recognition as sr
        
        # Создаем распознаватель с разными настройками
        r = sr.Recognizer()
        
        print("📊 Текущие настройки:")
        print(f"   - energy_threshold: {r.energy_threshold}")
        print(f"   - pause_threshold: {r.pause_threshold}")
        print(f"   - phrase_threshold: {r.phrase_threshold}")
        print(f"   - non_speaking_duration: {r.non_speaking_duration}")
        
        # Тестируем с микрофоном
        with sr.Microphone() as source:
            print("\n🎤 Калибровка микрофона...")
            print("📢 Помолчите 3 секунды...")
            r.adjust_for_ambient_noise(source, duration=3.0)
            
            print("\n📢 Теперь скажите короткую фразу...")
            print("⏱️ Ждите завершения фразы...")
            
            try:
                audio = r.listen(source, timeout=10, phrase_time_limit=15)
                text = r.recognize_google(audio, language='ru-RU')
                print(f"✅ Распознано: '{text}'")
                return True
            except sr.UnknownValueError:
                print("❌ Не удалось распознать речь")
                return False
            except sr.RequestError as e:
                print(f"❌ Ошибка сервиса: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Запуск тестов микрофона\n")
    
    # Тест настроек
    settings_ok = test_speech_recognition()
    
    # Тест микрофона
    mic_ok = test_microphone_settings()
    
    print(f"\n📊 Результаты тестирования:")
    print(f"   - Настройки распознавания: {'✅' if settings_ok else '❌'}")
    print(f"   - Микрофон: {'✅' if mic_ok else '❌'}")
    
    if settings_ok and mic_ok:
        print("\n🎉 Все тесты пройдены! Микрофон готов к работе.")
        print("💡 Совет: Говорите четко и делайте паузы между словами")
    else:
        print("\n❌ Обнаружены проблемы с микрофоном.")
        print("💡 Проверьте:")
        print("   - Подключен ли микрофон")
        print("   - Разрешения на доступ к микрофону")
        print("   - Интернет-соединение (для Google Speech Recognition)")
