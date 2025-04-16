
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
