# Кредитный скоринг с Telegram-ботом

## 📌 Описание проекта

Система кредитного скоринга на основе модели машинного обучения **XGBoost**, доступная через Telegram-бота. Анализирует параметры клиента, предсказывает вероятность одобрения кредита и сохраняет историю запросов в базе данных **PostgreSQL**.

---

## 🚀 Особенности проекта

- 📊 Анализ кредитоспособности клиентов по 20 параметрам
- 🤖 Модель XGBoost с оптимизированным порогом принятия решений
- 📉 Адаптивная оценка риска клиента
- 💬 Telegram-бот для получения предсказаний
- 🗃 Хранение данных и результатов в PostgreSQL
- 📈 Визуализация важных признаков
- 📊 Статистика по принятым решениям

---

## 🗂 Структура проекта

| Файл                         | Описание                                         |
|-----------------------------|--------------------------------------------------|
| `creditscoring.ipynb`       | Анализ данных и обучение модели (Jupyter)       |
| `telegram_bot.py`           | Реализация Telegram-бота                         |
| `german.data-numeric`       | Набор данных German Credit Data                 |
| `german_credit_xgboost_final.pkl` | Обученная модель с метаданными           |

---

## ⚙️ Технические требования

```bash
python >= 3.8
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
psycopg2
python-telegram-bot >= 13.0
sqlalchemy
imblearn
```

---

## 🛠 Установка и настройка

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/yourusername/telegramxcredit.git
cd telegramxcredit
```

2. **Создайте виртуальное окружение и установите зависимости:**

```bash
python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate     # Для Windows
pip install -r requirements.txt
```

3. **Настройте подключение к PostgreSQL в `telegram_bot.py`:**

```python
DB_CONFIG = {
  "dbname": "bank_data",
  "user": "your_username",
  "password": "your_password",
  "host": "localhost",
  "port": "5432"
}
```

4. **Создайте базу данных:**

```sql
CREATE DATABASE bank_data;
```

5. **Замените токен Telegram-бота:**

```python
token = 'your_telegram_bot_token'  # Получите у @BotFather в Telegram
```

6. **Запустите бота:**

```bash
python telegram_bot.py
```

---

## 💬 Использование Telegram-бота

**Доступные команды:**

- `/start` – Начало работы
- `/predict <параметры>` – Оценка кредитоспособности
- `/stats` – Статистика предсказаний
- `/importance` – Важность признаков модели
- `/compare [good|bad|average]` – Примеры клиентов
- `/help` – Помощь

**Пример запроса:**
```
/predict 1, 4, 12, 2, 0, 4000, 4, 5, 1, 1, 40, 1, 1, 1, 1, 4, 1, 1, 1, 1
```

---

## 🧠 Описание модели

Модель XGBoost:

- Балансировка классов через SMOTE
- Подбор гиперпараметров через GridSearchCV
- Оптимизация по F1-мере
- Добавлена оценка риска для принятия решений

---

## 🏷 Описание признаков

Модель использует 20 признаков, включая:

- `status` – статус счёта
- `duration` – срок кредита
- `credit_history` – история кредита
- `purpose` – цель кредита
- `credit_amount` – сумма
- `savings`, `employment`, `installment_rate`, `personal_status_sex` и др.

Полный список включён в разделе проекта.

---

## 🗄 Работа с базой данных

Используется **PostgreSQL**:

- Хранение предсказаний
- Хранение клиентских данных
- Статистика использования

**Таблица `credit_scoring_predictions` включает:**

- Входные параметры
- Рассчитанный риск
- Результат предсказания (одобрен/отклонён)
- Вероятность
- Дата и время

---

## 📚 Источник данных

**German Credit Data** — содержит 1000 заявок с метками "хороший"/"плохой" кредит. Широко используется в задачах кредитного скоринга.

---

## 🔧 Дополнительные возможности

- Расчёт факторов риска
- Адаптивный порог решений
- Генерация объяснений предсказаний
- Расчёт платежей
- История запросов для анализа

---

MIT License

Copyright (c) 2025 oTTomator and Archon contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.