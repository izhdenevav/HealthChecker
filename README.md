# 💻 HealthChecker — онлайн-анализ пульса и дыхания по видео

**HealthChecker** — это веб-приложение на Django, которое в режиме реального времени анализирует видеопоток с веб-камеры пользователя и извлекает ключевые физиологические показатели: **пульс (ЧСС)**, **дыхание** и **положение головы**.
Все данные можно сохранять в личном кабинете и отслеживать динамику.

https://izhdenevav-ml-project-54c6.twc1.net/
---
# Ветки

- docker - докерфайл и инструкция по сборке образа
- vlada - модуль определения пульса
- arina - модуль определения положения и поворота головы
- arseniy - модуль определения показателей дыхания
- lera - исходники веб-приложения
---

## 🚀 Основные возможности

- 📷 Получение видеопотока с веб-камеры через WebSocket
- 🧠 Анализ положения головы — определение корректного положения для съёмки
- ❤️ Извлечение пульса по лицу с помощью **FastBVP-Net**
- 🌬️ Анализ дыхательных колебаний
- 📈 Отображение графиков ЧСС и дыхания в реальном времени
- 👤 Система авторизации и сохранения измерений
- 🗃️ Просмотр истории измерений для зарегистрированных пользователей

---

## 🔐 Авторизация

- Нерегистрированные пользователи могут видеть пульс и дыхание, но данные не сохраняются.
- Авторизованные пользователи получают доступ к своей истории анализов.
- Поддерживается система пользователей Django (регистрация, вход, выход).

---

## 🧠 Алгоритмы в деталях

- FastBVP-Net — CNN-архитектура, обученная на UBFC-rPPG и собственном датасете. Извлекает сигнал ЧСС из видеоизменений на лице.
- Дыхание — анализируется по микроизменениям цвета лица.
- Положение головы — использует keypoint-детекцию (с помощью Mediapie) для определения допустимого угла наклона/поворота головы.

---

## 🧠 Используемые технологии


| Компонент                           | Технология                      |
| -------------------------------------------- | ----------------------------------------- |
| Бэкенд                                 | Django 5.x                                |
| Передача видеопотока      | WebSocket (JavaScript → Django Channels) |
| Модуль ЧСС                          | FastBVP-Net (PyTorch)                     |
| Модуль дыхания                  | OpenCV / PyTorch                          |
| Модуль положения головы | OpenCV / Mediapipe                        |
| Интерфейс                           | HTML, CSS, JS                             |
| База данных                        | PostgreSQL                                |

---

## 📦 Установка и запуск

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

### 4. Примените миграции:

```sh
python manage.py migrate
```

## Запустите проект:

После установки зависимостей можно запускать проект:

```sh
python manage.py runserver
```

## 📬 Обратная связь

### 📧 Telegram: https://t.me/+gyaVpbQfOrw0MmU6

---

🧠 Проект — студенческая исследовательская работа, открытая для развития и доработки.
