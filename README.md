=======
# ОПРЕДЕЛЕНИЕ ЧСС И ПОКАЗАТЕЛЕЙ ДЫХАНИЯ С УЧЕТОМ ПОВОРОТА ГОЛОВЫ

## Установка и запуск

### 1. Клонируйте репозиторий:
   ```sh
   git clone https://github.com/izhdenevav/ml_project.git
   cd ml_project
   ```

### 2. Создайте виртуальное окружение:

```sh
python -m venv venv
```

### 3. Активируйте окружение:

#### Windows:
```sh
venv\Scripts\activate
```

#### Linux/macOS:
```sh
source venv/bin/activate
```

### 3. Установите зависимости:

```sh
pip install -r requirements.txt
```

## Запустите проект:

После установки зависимостей можно запускать проект:

```sh
python main.py
```

## Дополнительно

- Для выхода из виртуального окружения используйте команду:
  ```sh
  deactivate
  ```
- Если в проект добавляются новые зависимости, обновите `requirements.txt` командой:
  ```sh
  pip freeze > requirements.txt
