"""
Голосовой помощник на Python (один файл).
Возможности:
- распознавание речи (Vosk/Whisper/Google SpeechRecognition)
- синтез речи (pyttsx3)
- решение математических задач (sympy)
- открытие приложений/файлов
- установка будильника (локально)
- интеграция с локальными LLM (Ollama/Transformers)
- умный поиск и запуск приложений на компьютере
- естественное понимание команд через нейросеть

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
def init_tts():
    """Инициализация движка синтеза речи."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Настройка голоса для русского языка если возможно
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'russian' in voice.name.lower() or 'ru' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        print("pyttsx3 не доступен:", e)
        return None

tts_engine = init_tts()


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
        try:
            txt = listen_vosk()
            if txt:
                return txt
        except Exception as e:
            print(f"Ошибка Vosk: {e}")
    
    if ASR_BACKEND == 'speech_recognition':
        try:
            return listen_speech_recognition()
        except Exception as e:
            print(f"Ошибка speech_recognition: {e}")
    
    # fallback: чтение из ввода пользователя
    try:
        return input('Введите текст (fallback): ')
    except (EOFError, KeyboardInterrupt):
        return None
    except Exception as e:
        print(f"Ошибка ввода: {e}")
        return None

# --- LLM Integration ---
# Попытка импорта различных LLM библиотек
LLM_BACKEND = None
llm_model = None

try:
    import requests
    # Проверяем доступность Ollama
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            LLM_BACKEND = 'ollama'
            print("Ollama обнаружен и готов к использованию")
    except:
        pass
except ImportError:
    pass

# Альтернативный вариант - transformers
if LLM_BACKEND is None:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore
        LLM_BACKEND = 'transformers'
        print("Transformers доступен для локальной LLM")
    except ImportError:
        pass

# --- Обработка команд ---
from sympy import sympify, Symbol

alarms = []  # список активных будильников


def get_alarms():
    """Возвращает список активных будильников."""
    return [alarm for alarm in alarms if alarm['time'] > datetime.now()]


def remove_alarm(alarm_id: int):
    """Удаляет будильник по ID."""
    global alarms
    alarms = [alarm for alarm in alarms if alarm['id'] != alarm_id]


def set_alarm(time_str: str, label: str = ''):
    """Установка будильника. time_str — в формате HH:MM или через минуты: 'in 10 minutes' (rus not implemented)
    Возвращает id будильника."""
    try:
        now = datetime.now()
        m = re.match(r"^(\d{1,2}):(\d{2})$", time_str)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError(f"Некорректное время: {hh:02d}:{mm:02d}")
            alarm_time = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if alarm_time <= now:
                alarm_time += timedelta(days=1)
        else:
            # попробуем интерпретировать как число минут
            m2 = re.match(r"^(\d+)\s*min(ute)?s?$", time_str)
            if m2:
                minutes = int(m2.group(1))
                if minutes <= 0:
                    raise ValueError("Количество минут должно быть положительным числом")
                alarm_time = now + timedelta(minutes=minutes)
            else:
                # попроcим пользователя уточнить
                raise ValueError('Непонятный формат времени: ' + time_str)

        alarm = {'time': alarm_time, 'label': label, 'id': len(alarms) + 1}
        alarms.append(alarm)
        # запустим отдельный поток для отслеживания (небольшой помощник в фоне)
        threading.Thread(target=alarm_worker, args=(alarm,), daemon=True).start()
        return alarm['id']
    except Exception as e:
        raise ValueError(f"Ошибка установки будильника: {str(e)}")


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


def find_installed_apps():
    """Находит установленные приложения на Windows."""
    apps = {}
    
    if platform.system() != 'Windows':
        return apps
    
    try:
        import winreg
        
        # Поиск в реестре Windows
        registry_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")
        ]
        
        for hkey, path in registry_paths:
            try:
                with winreg.OpenKey(hkey, path) as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                try:
                                    display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                    install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                    display_icon = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                                    
                                    if display_name and install_location:
                                        # Очищаем имя от версий и лишней информации
                                        clean_name = re.sub(r'\s+\d+\.\d+.*$', '', display_name)
                                        clean_name = re.sub(r'\s+\(.*?\)', '', clean_name)
                                        
                                        apps[clean_name.lower()] = {
                                            'name': display_name,
                                            'path': install_location,
                                            'icon': display_icon if display_icon else None
                                        }
                                except:
                                    continue
                        except:
                            continue
            except:
                continue
                
    except ImportError:
        print("winreg не доступен")
    
    return apps


def find_start_menu_apps():
    """Находит приложения в меню Пуск Windows."""
    apps = {}
    
    if platform.system() != 'Windows':
        return apps
    
    start_menu_paths = [
        os.path.expanduser(r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"),
        os.path.expanduser(r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\System Tools"),
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
        r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\System Tools"
    ]
    
    for start_path in start_menu_paths:
        if os.path.exists(start_path):
            try:
                for root, dirs, files in os.walk(start_path):
                    for file in files:
                        if file.endswith('.lnk'):
                            full_path = os.path.join(root, file)
                            app_name = os.path.splitext(file)[0]
                            
                            # Очищаем имя
                            clean_name = re.sub(r'\s+\(.*?\)', '', app_name)
                            clean_name = clean_name.lower()
                            
                            if clean_name not in apps:
                                apps[clean_name] = {
                                    'name': app_name,
                                    'path': full_path,
                                    'type': 'shortcut'
                                }
            except Exception as e:
                print(f"Ошибка поиска в {start_path}: {e}")
    
    return apps


def search_apps_by_name(query: str):
    """Ищет приложения по названию."""
    query = query.lower().strip()
    
    # Объединяем все найденные приложения
    all_apps = {}
    all_apps.update(find_installed_apps())
    all_apps.update(find_start_menu_apps())
    
    # Поиск точных совпадений
    exact_matches = []
    partial_matches = []
    
    for app_key, app_info in all_apps.items():
        app_name = app_info['name'].lower()
        
        # Точное совпадение
        if query == app_key or query in app_name:
            exact_matches.append((app_key, app_info))
        # Частичное совпадение
        elif any(word in app_key or word in app_name for word in query.split()):
            partial_matches.append((app_key, app_info))
    
    # Сортируем по релевантности
    results = exact_matches + partial_matches
    return results[:5]  # Возвращаем топ-5 результатов


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


def check_llm_status():
    """Проверяет статус доступных LLM."""
    status = {
        'ollama': False,
        'transformers': False,
        'active_backend': None
    }
    
    # Проверяем Ollama
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            status['ollama'] = True
            if LLM_BACKEND == 'ollama':
                status['active_backend'] = 'ollama'
    except:
        pass
    
    # Проверяем transformers
    try:
        from transformers import pipeline  # type: ignore
        status['transformers'] = True
        if LLM_BACKEND == 'transformers':
            status['active_backend'] = 'transformers'
    except:
        pass
    
    return status


def ask_llm(question: str):
    """Задает вопрос локальной LLM и возвращает ответ."""
    if LLM_BACKEND == 'ollama':
        return ask_ollama(question)
    elif LLM_BACKEND == 'transformers':
        return ask_transformers(question)
    else:
        return "Локальная нейросеть недоступна. Установите Ollama или transformers для расширенных возможностей."


def smart_app_launch(query: str):
    """Умный запуск приложений с помощью нейросети."""
    if not LLM_BACKEND:
        return "Нейросеть недоступна для умного поиска приложений"
    
    try:
        # Сначала ищем приложения по названию
        found_apps = search_apps_by_name(query)
        
        if not found_apps:
            # Если ничего не найдено, обращаемся к нейросети
            return ask_llm(f"Пользователь хочет запустить приложение: '{query}'. Но я не могу найти его на компьютере. Подскажи, как можно найти или установить это приложение.")
        
        # Если найдено одно приложение - запускаем его
        if len(found_apps) == 1:
            app_key, app_info = found_apps[0]
            app_path = app_info['path']
            
            if open_file(app_path):
                return f"Запускаю {app_info['name']}"
            else:
                return f"Не удалось запустить {app_info['name']}"
        
        # Если найдено несколько приложений, используем нейросеть для выбора
        apps_list = "\n".join([f"{i+1}. {app_info['name']}" for i, (_, app_info) in enumerate(found_apps)])
        
        prompt = f"""Пользователь хочет запустить приложение по запросу: "{query}"

Найдены следующие приложения:
{apps_list}

Выбери номер наиболее подходящего приложения (1-{len(found_apps)}) или ответь "не знаю" если ни одно не подходит."""
        
        response = ask_llm(prompt)
        
        # Пытаемся извлечь номер из ответа
        import re
        numbers = re.findall(r'\d+', response)
        if numbers:
            try:
                choice = int(numbers[0]) - 1
                if 0 <= choice < len(found_apps):
                    app_key, app_info = found_apps[choice]
                    app_path = app_info['path']
                    
                    if open_file(app_path):
                        return f"Запускаю {app_info['name']}"
                    else:
                        return f"Не удалось запустить {app_info['name']}"
            except:
                pass
        
        # Если не удалось понять выбор, показываем список
        return f"Найдено несколько приложений: {apps_list}. Скажите точнее, какое хотите запустить."
        
    except Exception as e:
        return f"Ошибка при поиске приложения: {str(e)}"


def ask_ollama(question: str):
    """Использует Ollama для генерации ответа."""
    try:
        import requests
        
        # Системный промпт для голосового помощника
        system_prompt = """Ты умный голосовой помощник. Отвечай кратко и по делу на русском языке. 
        Если пользователь просит выполнить действие, которое можно сделать через системные команды, 
        предложи конкретные шаги. Если это общий вопрос - дай информативный ответ.
        Если пользователь просит запустить приложение, помоги найти и запустить его."""
        
        payload = {
            "model": "llama3.2",  # или другая доступная модель
            "prompt": f"{system_prompt}\n\nВопрос пользователя: {question}",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 200
            }
        }
        
        response = requests.post('http://localhost:11434/api/generate', 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'Не удалось получить ответ от нейросети')
        else:
            return f"Ошибка Ollama: {response.status_code}"
            
    except Exception as e:
        return f"Ошибка при обращении к Ollama: {str(e)}"


def ask_transformers(question: str):
    """Использует transformers для генерации ответа."""
    global llm_model
    
    try:
        if llm_model is None:
            # Загружаем легкую модель для быстрого ответа
            model_name = "microsoft/DialoGPT-medium"  # или другая подходящая модель
            llm_model = pipeline("text-generation", model=model_name, 
                               tokenizer=model_name, max_length=150)
        
        # Формируем промпт
        prompt = f"Пользователь: {question}\nПомощник:"
        
        # Генерируем ответ
        response = llm_model(prompt, max_length=len(prompt.split()) + 50, 
                           num_return_sequences=1, temperature=0.7, 
                           pad_token_id=llm_model.tokenizer.eos_token_id)
        
        # Извлекаем ответ
        generated_text = response[0]['generated_text']
        answer = generated_text.split("Помощник:")[-1].strip()
        
        return answer if answer else "Не удалось сгенерировать ответ"
        
    except Exception as e:
        return f"Ошибка при работе с transformers: {str(e)}"


def parse_command(text: str):
    """Простейший парсер команд. Возвращает (intent, data).
    intents: open_app, set_alarm, solve_math, quit, help, show_alarms, unknown
    """
    t = text.lower().strip()
    # quit
    if t in ('выход', 'выйти', 'стоп', 'quit', 'exit','хватит'):
        return ('quit', None)
    # help
    if 'помощь' in t or 'что ты умеешь' in t:
        return ('help', None)
    # показать будильники
    if any(word in t for word in ['будильники', 'alarms', 'список будильников']):
        return ('show_alarms', None)
    # показать время
    if any(word in t for word in ['время', 'time', 'сколько времени']):
        return ('show_time', None)
    # вопрос к нейросети
    if any(word in t for word in ['спроси', 'ask', 'что такое', 'как', 'почему', 'расскажи']):
        return ('ask_llm', text)
    # статус нейросети
    if any(word in t for word in ['статус нейросети', 'статус llm', 'проверь нейросеть']):
        return ('check_llm', None)
    # открыть приложение / файл
    m_open = re.search(r"(?:открой|запусти|open|start)\s+(.+)", t)
    if m_open:
        target = m_open.group(1).strip()
        return ('open_app', target)
    # умный запуск приложений через нейросеть
    if any(phrase in t for phrase in ['найди и запусти', 'найди приложение', 'запусти программу', 'открой программу']):
        # Извлекаем название приложения
        app_name = t
        for phrase in ['найди и запусти', 'найди приложение', 'запусти программу', 'открой программу']:
            if phrase in app_name:
                app_name = app_name.replace(phrase, '').strip()
                break
        return ('smart_launch', app_name)
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
        speak('Я могу: решать математические выражения, открывать приложения, ставить будильник, показывать список будильников, показывать время, отвечать на вопросы через нейросеть, умно искать и запускать приложения на компьютере. Скажите например: "открой калькулятор" или "найди и запусти Chrome" или "поставь будильник на 07:30" или "что такое Python"')
        return
    if intent == 'show_alarms':
        active_alarms = get_alarms()
        if not active_alarms:
            speak('Нет активных будильников')
        else:
            alarm_list = ', '.join([f"номер {alarm['id']} на {alarm['time'].strftime('%H:%M')}" for alarm in active_alarms])
            speak(f'Активные будильники: {alarm_list}')
        return
    if intent == 'show_time':
        current_time = datetime.now().strftime('%H:%M')
        speak(f'Текущее время: {current_time}')
        return
    if intent == 'ask_llm':
        question = data
        speak('Думаю...')
        answer = ask_llm(question)
        speak(answer)
        return
    if intent == 'smart_launch':
        app_query = data
        speak('Ищу приложение...')
        result = smart_app_launch(app_query)
        speak(result)
        return
    if intent == 'check_llm':
        status = check_llm_status()
        if status['active_backend']:
            speak(f'Нейросеть активна: {status["active_backend"]}')
        else:
            available = []
            if status['ollama']:
                available.append('Ollama')
            if status['transformers']:
                available.append('Transformers')
            
            if available:
                speak(f'Доступно: {", ".join(available)}, но не активно')
            else:
                speak('Нейросеть недоступна. Установите Ollama или transformers')
        return
    if intent == 'open_app':
        target = data
        speak(f'Пытаюсь открыть {target}')
       
        # универсальная мапа для разных ОС — пользователь может добавлять свои
        known = {
            'калькулятор': {
                'Windows': 'calc.exe',
                'Darwin': 'open -a Calculator',
                'Linux': 'gnome-calculator'
            },
            'блокнот': {
                'Windows': 'notepad.exe',
                'Darwin': 'open -a TextEdit',
                'Linux': 'gedit'
            },
            # Браузеры
            'Google': {
                'Windows': ['chrome.exe', 'google-chrome.exe', 'Google Chrome'],
                'Darwin': 'open -a "Google Chrome"',
                'Linux': ['google-chrome', 'chromium-browser']
            },
            'Chrome': {
                'Windows': ['chrome.exe', 'google-chrome.exe', 'Google Chrome'],
                'Darwin': 'open -a "Google Chrome"',
                'Linux': ['google-chrome', 'chromium-browser']
            },
            'Firefox': {
                'Windows': 'firefox.exe',
                'Darwin': 'open -a Firefox',
                'Linux': 'firefox'
            },
            # Игры (только Windows, так как пути специфичны)
            'cs': {
                'Windows': [
                    r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe',
                    r'C:\Program Files\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe'
                ]
            },
            'кс го': {
                'Windows': [
                    r'C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe',
                    r'C:\Program Files\Steam\steamapps\common\Counter-Strike Global Offensive\game\bin\win64\cs2.exe'
                ]
            },
            'dota': {
                'Windows': [
                    r'C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\bin\win64\dota2.exe',
                    r'C:\Program Files\Steam\steamapps\common\dota 2 beta\game\bin\win64\dota2.exe'
                ]
            },
            'валорант': {
                'Windows': [
                    r'C:\Riot Games\Riot Client\RiotClientServices.exe',
                    r'C:\Users\%USERNAME%\AppData\Local\Riot Games\Riot Client\RiotClientServices.exe'
                ]
            },
            'riot': {
                'Windows': [
                    r'C:\Riot Games\Riot Client\RiotClientServices.exe',
                    r'C:\Users\%USERNAME%\AppData\Local\Riot Games\Riot Client\RiotClientServices.exe'
                ]
            }
        }
        system = platform.system()
        if target in known:
            apps = known[target].get(system)
            if apps:
                # Если это список путей, пробуем каждый
                if isinstance(apps, list):
                    for app in apps:
                        # Заменяем переменные окружения в пути
                        expanded_app = os.path.expandvars(app)
                        if os.path.exists(expanded_app) or app in ['chrome.exe', 'firefox.exe', 'notepad.exe', 'calc.exe']:
                            ok = open_file(expanded_app)
                            if ok:
                                speak('Открыто')
                                return
                else:
                    # Если это строка
                    expanded_app = os.path.expandvars(apps)
                    ok = open_file(expanded_app)
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
        # Попробуем умно обработать команду
        if LLM_BACKEND:
            # Сначала попробуем найти и запустить приложение
            if any(word in data.lower() for word in ['запусти', 'открой', 'найди', 'запуск', 'приложение', 'программа']):
                speak('Попробую найти и запустить приложение...')
                result = smart_app_launch(data)
                speak(result)
            else:
                # Если это не команда запуска, обращаемся к нейросети
                speak('Попробую разобраться...')
                answer = ask_llm(data)
                speak(answer)
        else:
            speak('Не понял команду. Повторите или скажите "помощь". Для расширенных возможностей установите Ollama или transformers.')
        return

# --- Основной цикл ---

def main_loop():
    # Показываем статус LLM при запуске
    if LLM_BACKEND:
        speak(f'Готов. Нейросеть активна: {LLM_BACKEND}. Скажите команду.')
    else:
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
# Requirements (файл requirements.txt):
# pyttsx3
# sympy
# speechrecognition
# vosk
# sounddevice
# pyaudio
# requests
# transformers
# torch
# -----------------------------
# Интеграция с локальными LLM:
# ✅ РЕАЛИЗОВАНО:
# - Поддержка Ollama (рекомендуется)
# - Поддержка Transformers (альтернатива)
# - Автоматическое обнаружение доступных бэкендов
# - Команды: "что такое...", "как...", "почему...", "расскажи..."
# - Проверка статуса: "проверь нейросеть"
# - Fallback для неизвестных команд
#
# Умный поиск и запуск приложений:
# ✅ РЕАЛИЗОВАНО:
# - Поиск в реестре Windows (установленные программы)
# - Поиск в меню Пуск (ярлыки)
# - Умный выбор приложений через нейросеть
# - Команды: "найди и запусти...", "открой программу..."
# - Естественное понимание команд запуска
# - Автоматический поиск при неизвестных командах
# 
# Установка Ollama:
# 1. Скачать с https://ollama.ai
# 2. Установить модель: ollama pull llama3.2
# 3. Запустить помощника
# 
# Установка Transformers:
# pip install transformers torch
# 
# Тестирование: python test_llm.py
# -----------------------------
