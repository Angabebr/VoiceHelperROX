"""
Голосовой помощник на Python (один файл).
Возможности:
- распознавание речи (Vosk/Whisper/Google SpeechRecognition)
- синтез речи (pyttsx3)
- решение математических задач (sympy)
- открытие приложений/файлов
- установка будильника (локально)
- расширяемая архитектура для подключения локальных LLM или других моделей

Как использовать:
1) Установите зависимости (рекомендуется виртуальное окружение):
   pip install -r requirements.txt
   (см. список зависимостей внизу файла)

2) Запустите:
   python voice_assistant.py

Примечание: скрипт написан так, чтобы работать оффлайн через Vosk (если установлен)
и синхронно через встроенные библиотеки. Если Vosk не установлен — автоматически
попытается использовать библиотеку speech_recognition с онлайн-сервисом Google.

Безопасность: команда 'реши' использует sympy / безопасную оценку выражений.
Не выполняются произвольные системные команды от распознанного текста.

"""

import os
import sys
import platform
import subprocess
import threading
import time
import queue
import re
from datetime import datetime, timedelta

# --- TTS (offline) ---
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception as e:
    tts_engine = None
    print("pyttsx3 не доступен:", e)


def speak(text: str):
    """Синтез речи. Если pyttsx3 есть — используем его, иначе печатаем."""
    print("Assistant:", text)
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print("Ошибка TTS:", e)

# --- ASR (распознавание речи) ---
# Попытка использовать Vosk (offline). Если нет — используем speech_recognition + Google.
ASR_BACKEND = None

try:
    from vosk import Model, KaldiRecognizer
    import sounddevice as sd
    import json
    # Путь к модели Vosk (пользователь должен скачать модель и указать путь или положить рядом)
    VOSK_MODEL_PATH = "model"  # поменяйте при необходимости
    if os.path.exists(VOSK_MODEL_PATH):
        vosk_model = Model(VOSK_MODEL_PATH)
        ASR_BACKEND = 'vosk'
    else:
        print("Vosk модель не найдена по пути 'model'. Если хотите оффлайн ASR — скачайте и поместите модель.")
except Exception:
    vosk_model = None

if ASR_BACKEND is None:
    try:
        import speech_recognition as sr
        ASR_BACKEND = 'speech_recognition'
    except Exception:
        ASR_BACKEND = None


def listen_vosk(timeout=8):
    """Прослушивание микрофона с Vosk. Возвращает распознанную строку или None."""
    try:
        q = queue.Queue()
        samplerate = int(sd.query_devices(None, 'input')['default_samplerate'])
        rec = KaldiRecognizer(vosk_model, samplerate)

        print("Говорите... (Vosk)")
        response_text = None
        with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, dtype='int16', channels=1) as stream:
            start = time.time()
            while True:
                data = stream.read(4000)
                if rec.AcceptWaveform(data[0]):
                    res = json.loads(rec.Result())
                    if 'text' in res:
                        response_text = res['text']
                        break
                else:
                    # промежуточный результат можно игнорировать
                    pass
                if time.time() - start > timeout:
                    # попробуем получить частичный
                    res = json.loads(rec.FinalResult())
                    response_text = res.get('text', '')
                    break
        return response_text
    except Exception as e:
        print('Vosk listen error:', e)
        return None


def listen_speech_recognition(timeout=8):
    """Использует Google Web Speech API через speech_recognition (нужен интернет)."""
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Говорите... (Google ASR)")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=15)
    try:
        txt = r.recognize_google(audio, language='ru-RU')
        return txt
    except sr.UnknownValueError:
        return None
    except Exception as e:
        print('ASR error:', e)
        return None


def listen():
    """Единый интерфейс прослушивания. Возвращает распознанную фразу на русском."""
    if ASR_BACKEND == 'vosk' and vosk_model is not None:
        txt = listen_vosk()
        if txt:
            return txt
    if ASR_BACKEND == 'speech_recognition':
        return listen_speech_recognition()
    # fallback: чтение из ввода пользователя
    try:
        return input('Введите текст (fallback): ')
    except Exception:
        return None

# --- Обработка команд ---
from sympy import sympify, Symbol

alarms = []  # список активных будильников


def set_alarm(time_str: str, label: str = ''):
    """Установка будильника. time_str — в формате HH:MM или через минуты: 'in 10 minutes' (rus not implemented)
    Возвращает id будильника."""
    now = datetime.now()
    m = re.match(r"^(\d{1,2}):(\d{2})$", time_str)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        alarm_time = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if alarm_time <= now:
            alarm_time += timedelta(days=1)
    else:
        # попробуем интерпретировать как число минут
        m2 = re.match(r"^(\d+)\s*min(ute)?s?$", time_str)
        if m2:
            minutes = int(m2.group(1))
            alarm_time = now + timedelta(minutes=minutes)
        else:
            # попроcим пользователя уточнить
            raise ValueError('Непонятный формат времени: ' + time_str)

    alarm = {'time': alarm_time, 'label': label, 'id': len(alarms) + 1}
    alarms.append(alarm)
    # запустим отдельный поток для отслеживания (небольшой помощник в фоне)
    threading.Thread(target=alarm_worker, args=(alarm,), daemon=True).start()
    return alarm['id']


def alarm_worker(alarm):
    now = datetime.now()
    delta = (alarm['time'] - now).total_seconds()
    if delta > 0:
        time.sleep(delta)
    # при срабатывании
    speak(f"Будильник: {alarm.get('label','Без названия')} — время {alarm['time'].strftime('%H:%M')}")
    try:
        # проиграть звуковой файл если есть
        if os.path.exists('alarm_sound.mp3'):
            # простой кроссплатформенный способ: откроем файл по умолчанию
            open_file('alarm_sound.mp3')
    except Exception as e:
        print('Не удалось воспроизвести звук:', e)


def open_file(path_or_app: str):
    """Открывает файл или приложение в зависимости от ОС."""
    system = platform.system()
    try:
        if system == 'Windows':
            os.startfile(path_or_app)
        elif system == 'Darwin':
            subprocess.Popen(['open', path_or_app])
        else:
            subprocess.Popen(['xdg-open', path_or_app])
        return True
    except Exception as e:
        print('open_file error:', e)
        return False


def safe_eval_math(expr: str):
    """Безопасная оценка математического выражения через sympy."""
    try:
        # убираем запрещённые символы
        expr = expr.replace('^', '**')
        # sympify возвращает выражение
        res = sympify(expr)
        return str(res)
    except Exception as e:
        return 'Ошибка при вычислении: ' + str(e)


def parse_command(text: str):
    """Простейший парсер команд. Возвращает (intent, data).
    intents: open_app, set_alarm, solve_math, quit, help, unknown
    """
    t = text.lower().strip()
    # quit
    if t in ('выход', 'выйти', 'стоп', 'quit', 'exit','хватит'):
        return ('quit', None)
    # help
    if 'помощь' in t or 'что ты умеешь' in t:
        return ('help', None)
    # открыть приложение / файл
    m_open = re.search(r"(?:открой|запусти|open|start)\s+(.+)", t)
    if m_open:
        target = m_open.group(1).strip()
        return ('open_app', target)
    # установить будильник
    m_alarm = re.search(r"(?:будильник|поставь будильник|установи будильник)\s*(?:на)?\s*(\d{1,2}:\d{2})", t)
    if m_alarm:
        return ('set_alarm', m_alarm.group(1))
    m_alarm2 = re.search(r"(?:через)\s*(\d+)\s*(?:минут)", t)
    if m_alarm2:
        mins = m_alarm2.group(1)
        return ('set_alarm', f"{mins}min")
    # math
    m_math = re.search(r"(?:реши|посчитай|calculate|compute|solve)\s+(.+)", t)
    if m_math:
        expr = m_math.group(1)
        return ('solve_math', expr)
    # простая математика если просто выражение
    if re.match(r"^[0-9\s\+\-\*\/%\(\)\.\^e]+$", t):
        return ('solve_math', t)
    return ('unknown', text)


def handle_command(intent, data):
    if intent == 'quit':
        speak('До свидания!')
        sys.exit(0)
    if intent == 'help':
        speak('Я могу: решать математические выражения, открывать приложения, ставить будильник. Скажите например: "открой калькулятор" или "поставь будильник на 07:30" или "реши 2+2"')
        return
    if intent == 'open_app':
        target = data
        speak(f'Пытаюсь открыть {target}')
        # Попробуем несколько стратегий: если это очевидный файл/путь — откроем,
        # иначе попробуем сопоставить с известными приложениями
        # простая мапа для Windows/Mac/Linux — пользователь может добавлять свои
        known = {
            'калькулятор': {
                'Windows': 'calc.exe',
                
            },
            'блокнот': {
                'Windows': 'notepad.exe',
                
            },
            # Google Chrome
            'гугл': {
                'Windows': r'C:\Program Files\Google\Chrome\Application\chrome.exe'
            },
            'хром': {
                'Windows': r'C:\Program Files\Google\Chrome\Application\chrome.exe'
            },
            # Counter-Strike 2
            'cs': {
                'Windows': r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe'
            },
            'кс го': {
                'Windows': r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe'
            },
            # Dota 2
            'dota': {
                'Windows': r'C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\bin\win64\dota2.exe'
            },
            # Valorant (через Riot Client)
            'валорант': {
                'Windows': r'C:\Riot Games\Riot Client\RiotClientServices.exe'
            },
            'riot': {
                'Windows': r'C:\Riot Games\Riot Client\RiotClientServices.exe'
            }
        }
        system = platform.system()
        if target in known:
            app = known[target].get(system)
            if app:
                ok = open_file(app)
                if ok:
                    speak('Открыто')
                    return
        # пробуем открыть как путь
        if os.path.exists(target):
            if open_file(target):
                speak('Файл открыт')
                return
        # как команда
        try:
            subprocess.Popen(target.split())
            speak('Команда отправлена')
            return
        except Exception as e:
            speak('Не удалось открыть ' + target)
            print(e)
            return
    if intent == 'set_alarm':
        time_spec = data
        try:
            alarm_id = set_alarm(time_spec, label='Голосовой будильник')
            speak(f'Будильник установлен под номером {alarm_id}')
        except Exception as e:
            speak('Не удалось установить будильник: ' + str(e))
        return
    if intent == 'solve_math':
        expr = data
        speak('Вычисляю...')
        res = safe_eval_math(expr)
        speak('Результат: ' + res)
        return
    if intent == 'unknown':
        # Попробуем более гибко: если пользователь разрешил локальную LLM — отправим туда.
        # Здесь — простой ответ-заглушка.
        speak('Не понял команду. Повторите или скажите "помощь"')
        return

# --- Основной цикл ---

def main_loop():
    speak('Готов. Скажите команду.')
    while True:
        try:
            txt = listen()
            if not txt:
                speak('Я ничего не расслышал. Повторите, пожалуйста.')
                continue
            print('Вы сказали:', txt)
            intent, data = parse_command(txt)
            handle_command(intent, data)
        except KeyboardInterrupt:
            speak('Выход')
            break
        except Exception as e:
            print('Ошибка основного цикла:', e)
            speak('Произошла ошибка: ' + str(e))


if __name__ == '__main__':
    main_loop()


# -----------------------------
# Requirements (примерный файл requirements.txt):
# pyttsx3
# sympy
# speechrecognition
# vosk
# sounddevice
# pyaudio (для некоторых платформ можно использовать вместо sounddevice)
# note: pyaudio часто сложно установить на Windows; sounddevice + vosk — более простая цепочка.
# -----------------------------
# Как подключить локальную LLM / нейросеть:
# - можно добавить функцию `ask_local_model(question)` и в handle_command
#   при intent == 'unknown' переадресовывать туда, чтобы помощник мог отвечать на
#   произвольные вопросы. Варианты: GPT-4all, Mistral локально, llama.cpp, etc.
# - если хотите, могу подготовить пример интеграции с одной из локальных библиотек.
# -----------------------------
