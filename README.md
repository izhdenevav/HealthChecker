<<<<<<< Updated upstream
# ml_project lera's branch

## Как запустить проект

```
python manage.py runserver
```

После запуска сервер будет доступен по адресу [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

---

## Краткое описание структуры проекта

### Изменения в файлах `urls.py`

- В папке **webheadpose** файл `urls.py` является основным.
- В нем добавлена строка:

  ```
  path('', include('videodelivery.urls')),
  ```

  Она перенаправляет все запросы по адресу [http://127.0.0.1:8000/](http://127.0.0.1:8000/) на файл `urls.py` внутри приложения **videodelivery**.

- В `urls.py` приложения **videodelivery** определены маршруты:

  ```
  path('', views.index, name='index'),
  path('video_feed/', views.video_feed, name='video_feed'),
  ```

  Это значит, что:
  
  - Переход на [http://127.0.0.1:8000/](http://127.0.0.1:8000/) вызывает функцию `index` из `views.py`.
  - Переход на [http://127.0.0.1:8000/video_feed](http://127.0.0.1:8000/video_feed) вызывает функцию `video_feed` из `views.py`.

### О функциях `index` и `video_feed`

- Эти функции находятся в `views.py` папки **videodelivery**.
- Реализованы на основе работы Арины и Арсения.

## Примечание

Остальные файлы приложния оставлены без изменений — они были автоматически созданы при создании проекта.
=======

# ОПРЕДЕЛЕНИЕ ЧСС

# Установка окружения и зависимостей

## 1. Клонирование репозитория (если требуется)
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## 2. Создание виртуального окружения

### Вариант A: через `venv` + `pip`
> Подходит, если у тебя установлен только `Python` и `pip`.

1. Создай виртуальное окружение:
   ```bash
   python -m venv venv
   ```

2. Активируй его:

   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

   - **Windows (cmd):**
     ```cmd
     venv\Scripts\activate
     ```

   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

3. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### Вариант B: через `conda`
> Подходит, если ты используешь `Anaconda` или `Miniconda`.

1. Создай окружение с нужной версией Python:
   ```bash
   conda create -n myenv python=3.11
   ```

2. Активируй окружение:
   ```bash
   conda activate myenv
   ```

3. Установи зависимости из `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Проверка

После установки ты можешь запустить свой скрипт, чтобы убедиться, что всё работает:

```bash
python main.py
```

Или другой файл, с которого начинается выполнение программы.
>>>>>>> Stashed changes
