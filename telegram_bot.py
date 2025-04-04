import logging
import pandas as pd
import psycopg2
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import pickle
import numpy as np
import os
from contextlib import contextmanager
from xgboost import XGBClassifier

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы
MODEL_PATH = 'german_credit_xgboost_final.pkl'  # Путь к модели XGBoost
OPTIMAL_THRESHOLD = 0.1935  # Оптимальный порог из нашего анализа
MODEL_VERSION = "XGBoost с оптимальным порогом"

# Параметры подключения к PostgreSQL
DB_CONFIG = {
    "dbname": "bank_data",
    "user": "sleep",
    "password": "",
    "host": "localhost",
    "port": "5432"
}

# Словари для расшифровки категориальных признаков
FEATURE_MAPPINGS = {
    'status': {
        1: '< 0 DM (отрицательный баланс)',
        2: 'нет чекового счета',
        3: '0 to 200 DM',
        4: '>= 200 DM (положительный баланс)'
    },
    'purpose': {
        0: 'новый автомобиль',
        1: 'подержанный автомобиль',
        2: 'мебель/оборудование',
        3: 'радио/телевизор',
        4: 'бытовая техника',
        5: 'ремонт',
        6: 'образование',
        7: 'отпуск',
        8: 'переобучение',
        9: 'бизнес',
        10: 'другое'
    },
    'credit_history': {
        0: 'нет кредитов/все погашены',
        1: 'все кредиты в банке погашены',
        2: 'существующие кредиты погашены',
        3: 'задержки в прошлом',
        4: 'критический счет/другие кредиты'
    },
    'savings': {
        1: '< 100 DM',
        2: '100-500 DM',
        3: '500-1000 DM',
        4: '>= 1000 DM',
        5: 'неизвестно/нет сбережений'
    },
    'employment': {
        1: 'безработный',
        2: '< 1 года',
        3: '1-4 года',
        4: '4-7 лет',
        5: '>= 7 лет'
    }
}

# Примеры клиентов для команды /compare
CLIENT_EXAMPLES = {
    "good": {
        'status': 4, 'duration': 12, 'credit_history': 2, 'purpose': 0, 'amount': 2000,
        'savings': 4, 'employment_duration': 5, 'installment_rate': 4, 'personal_status_sex': 3,
        'other_debtors': 1, 'residence_since': 4, 'property': 1, 'age': 40,
        'other_installment_plans': 1, 'housing': 1, 'existing_credits': 1, 'job': 4,
        'num_dependents': 1, 'telephone': 1, 'foreign_worker': 1
    },
    "bad": {
        'status': 1, 'duration': 60, 'credit_history': 4, 'purpose': 7, 'amount': 30000,
        'savings': 1, 'employment_duration': 1, 'installment_rate': 1, 'personal_status_sex': 2,
        'other_debtors': 3, 'residence_since': 1, 'property': 3, 'age': 19,
        'other_installment_plans': 3, 'housing': 3, 'existing_credits': 3, 'job': 1,
        'num_dependents': 2, 'telephone': 2, 'foreign_worker': 2
    },
    "average": {
        'status': 3, 'duration': 24, 'credit_history': 2, 'purpose': 2, 'amount': 8000,
        'savings': 3, 'employment_duration': 3, 'installment_rate': 2, 'personal_status_sex': 1,
        'other_debtors': 1, 'residence_since': 3, 'property': 2, 'age': 32,
        'other_installment_plans': 1, 'housing': 2, 'existing_credits': 1, 'job': 3,
        'num_dependents': 1, 'telephone': 1, 'foreign_worker': 1
    }
}

class CreditScoringBot:
    def __init__(self):
        """Инициализация бота для кредитного скоринга"""
        self.model = None
        self.scaler = None
        self.optimal_threshold = OPTIMAL_THRESHOLD
        self.feature_names = []
        self.load_model()
        self.ensure_db_table()

    def load_model(self):
        """Загрузка модели XGBoost"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as file:
                    model_data = pickle.load(file)
                    
                if isinstance(model_data, dict):  # Если модель сохранена в словаре с метаданными
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.optimal_threshold = model_data.get('optimal_threshold', OPTIMAL_THRESHOLD)
                    self.feature_names = model_data.get('required_features', [])
                else:  # Если сохранена только модель
                    self.model = model_data
                    
                logger.info(f"Модель {MODEL_VERSION} успешно загружена")
            else:
                logger.warning(f"Файл модели {MODEL_PATH} не найден. Создаем модель-заглушку.")
                self._create_dummy_model()
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            logger.info("Создаем модель-заглушку.")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Создание модели-заглушки"""
        self.model = XGBClassifier(objective='binary:logistic', random_state=42)
        
        # Создаем фиктивные данные для обучения
        X_dummy = np.random.rand(100, 20)
        y_dummy = np.random.randint(0, 2, 100)
        
        # Обучаем модель на фиктивных данных
        self.model.fit(X_dummy, y_dummy)
        
        self.optimal_threshold = OPTIMAL_THRESHOLD
        self.scaler = None
        
        # Задаем имена признаков для модели-заглушки
        self.feature_names = [
            'status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings',
            'employment', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'present_residence', 'property', 'age',
            'other_installment_plans', 'housing', 'existing_credits', 'job',
            'num_dependents', 'telephone', 'foreign_worker'
        ]
        
        logger.info("Модель-заглушка создана успешно")

    @contextmanager
    def get_db_connection(self):
        """Контекстный менеджер для работы с PostgreSQL"""
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def ensure_db_table(self):
        """Создание таблицы в PostgreSQL, если она не существует"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS credit_scoring_predictions (
                    id SERIAL PRIMARY KEY,
                    status INTEGER,
                    duration INTEGER,
                    credit_history INTEGER,
                    purpose INTEGER,
                    amount INTEGER,
                    savings INTEGER,
                    employment_duration INTEGER,
                    installment_rate INTEGER,
                    personal_status_sex INTEGER,
                    other_debtors INTEGER,
                    residence_since INTEGER,
                    property INTEGER,
                    age INTEGER,
                    other_installment_plans INTEGER,
                    housing INTEGER,
                    existing_credits INTEGER,
                    job INTEGER,
                    num_dependents INTEGER,
                    telephone INTEGER,
                    foreign_worker INTEGER,
                    risk_score INTEGER,
                    prediction TEXT,
                    probability FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                conn.commit()
                logger.info("Таблица credit_scoring_predictions готова к использованию")
        except Exception as e:
            logger.error(f"Ошибка при инициализации таблицы в БД: {e}")

    def preprocess_client_data(self, user_data):
        """Предобработка данных клиента для модели"""
        # Создаем копию данных пользователя
        processed_data = user_data.copy()
        
        # Преобразуем в DataFrame
        df = pd.DataFrame([processed_data]) if isinstance(processed_data, dict) else processed_data.copy()
        
        # Переименовываем поля для совместимости с моделью
        field_mapping = {
            'amount': 'credit_amount', 
            'employment_duration': 'employment',
            'residence_since': 'present_residence'
        }
        
        # Применяем маппинг
        for old_name, new_name in field_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Рассчитываем риск-фактор
        df['risk_score'] = (df['duration'] > 24).astype(int) + \
                          (df['credit_history'] >= 3).astype(int) * 2 + \
                          (df['status'] == 1).astype(int) * 2
                          
        credit_field = 'credit_amount' if 'credit_amount' in df.columns else 'amount'
        df['risk_score'] += (df[credit_field] > 10000).astype(int)
        
        employment_field = 'employment' if 'employment' in df.columns else 'employment_duration'
        df['risk_score'] += (df['age'] < 25).astype(int) + \
                          (df['savings'] <= 1).astype(int) + \
                          (df[employment_field] <= 2).astype(int) + \
                          (df['installment_rate'] == 1).astype(int)
        
        # Рассчитываем ежемесячный платеж
        if 'duration' in df.columns:
            if 'credit_amount' in df.columns:
                df['monthly_payment'] = df['credit_amount'] / df['duration']
            elif 'amount' in df.columns:
                df['monthly_payment'] = df['amount'] / df['duration']
        
        # Проверяем наличие всех необходимых полей для модели
        required_fields = set(self.feature_names)
        missing_fields = required_fields - set(df.columns)
        
        if missing_fields:
            logger.warning(f"Отсутствуют поля: {missing_fields}. Заполняем нулями.")
            for field in missing_fields:
                df[field] = 0
                
        return df

    async def save_data_to_db(self, user_data, prediction, prob, risk_score):
        """Сохранение данных в PostgreSQL"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = """
                INSERT INTO credit_scoring_predictions 
                (status, duration, credit_history, purpose, amount, savings, employment_duration,
                 installment_rate, personal_status_sex, other_debtors, residence_since, property,
                 age, other_installment_plans, housing, existing_credits, job, num_dependents,
                 telephone, foreign_worker, risk_score, prediction, probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Используем amount, а не credit_amount для базы данных
                amount_value = user_data.get('amount', user_data.get('credit_amount', 0))
                
                values = (
                    user_data['status'], user_data['duration'], user_data['credit_history'],
                    user_data['purpose'], amount_value, user_data['savings'],
                    user_data.get('employment_duration', user_data.get('employment', 0)), 
                    user_data['installment_rate'],
                    user_data['personal_status_sex'], user_data['other_debtors'],
                    user_data.get('residence_since', user_data.get('present_residence', 0)), 
                    user_data['property'], user_data['age'],
                    user_data['other_installment_plans'], user_data['housing'],
                    user_data['existing_credits'], user_data['job'], user_data['num_dependents'],
                    user_data['telephone'], user_data['foreign_worker'], risk_score,
                    prediction, prob
                )
                
                cursor.execute(query, values)
                conn.commit()
                logger.info("Данные успешно сохранены в PostgreSQL")
                return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении в БД: {e}")
            return False

    async def predict_credit_risk(self, user_data):
        """Предсказание кредитного риска с использованием модели XGBoost"""
        try:
            # Предобработка данных
            df_preprocessed = self.preprocess_client_data(user_data)
            
            # Сохраняем значение риск-фактора
            risk_score = int(df_preprocessed['risk_score'].values[0])
            
            # Выбираем только нужные для модели признаки
            if not self.feature_names:
                # Если имена признаков не заданы, используем все доступные
                model_features = [col for col in df_preprocessed.columns if col not in ['risk_score', 'monthly_payment']]
            else:
                # Используем заданные имена признаков
                model_features = self.feature_names
                
            # Проверяем, что все необходимые признаки есть в данных
            for feature in model_features:
                if feature not in df_preprocessed.columns:
                    raise ValueError(f"Отсутствует необходимый признак: {feature}")
            
            # Выбираем только признаки модели
            X = df_preprocessed[model_features]
            
            # Масштабирование данных, если есть скейлер
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Делаем предсказание
            prob = self.model.predict_proba(X_scaled)[:, 1]
            
            # Если риск-фактор очень высокий (>= 8), используем повышенный порог
            # Если клиент безработный (employment == 1) и сумма > 10000, автоматически отклоняем
            auto_reject = False
            adaptive_threshold = self.optimal_threshold
            
            if user_data.get('employment', user_data.get('employment_duration', 0)) == 1 and \
               user_data.get('credit_amount', user_data.get('amount', 0)) > 10000:
                auto_reject = True
                logger.info("Автоматическое отклонение: безработный клиент запрашивает большую сумму")
            elif risk_score >= 8:
                adaptive_threshold = max(0.7, self.optimal_threshold)
                logger.info(f"Высокий риск ({risk_score}/10), используем повышенный порог: {adaptive_threshold}")
            
            # Принимаем решение
            prediction = 0 if auto_reject else (prob >= adaptive_threshold).astype(int)[0]
            
            # Определяем текстовый результат
            prediction_label = "Одобрен" if prediction == 1 else "Отклонен"
            
            # Формируем результат
            result = {
                "prediction": prediction,
                "probability": float(prob[0]),
                "prediction_label": prediction_label,
                "risk_score": risk_score,
                "applied_threshold": adaptive_threshold,
                "auto_reject": auto_reject
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return {"error": str(e)}

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        await update.message.reply_text(
            f'Привет! Я бот для оценки вероятности одобрения кредита, использующий модель {MODEL_VERSION}.\n\n'
            'Используй команду /predict для оценки кредитоспособности.\n\n'
            'Пример: /predict 4, 12, 2, 0, 4000, 4, 5, 4, 1, 1, 4, 1, 40, 1, 1, 1, 4, 1, 1, 1\n\n'
            'Доступные команды:\n'
            '/stats - просмотр статистики предсказаний\n'
            '/importance - важность признаков для модели\n'
            '/compare - примеры клиентов\n'
            '/help - подробная информация о признаках'
        )

    async def predict_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /predict"""
        try:
            # Получаем данные от пользователя
            user_message = update.message.text
            if user_message.startswith('/predict'):
                user_message = user_message[8:].strip()
                
            data = [item.strip() for item in user_message.split(',')]
            
            if len(data) < 20:
                await update.message.reply_text("Недостаточно данных. Пожалуйста, предоставьте все 20 параметров.")
                return
            
            # Преобразуем данные в словарь
            try:
                user_data = {
                    'status': int(data[0]),
                    'duration': int(data[1]),
                    'credit_history': int(data[2]),
                    'purpose': int(data[3]),
                    'credit_amount': int(data[4]),
                    'amount': int(data[4]),  # Дублируем для совместимости
                    'savings': int(data[5]),
                    'employment': int(data[6]),  # Имя для модели
                    'employment_duration': int(data[6]),  # Имя для БД
                    'installment_rate': int(data[7]),
                    'personal_status_sex': int(data[8]),
                    'other_debtors': int(data[9]),
                    'present_residence': int(data[10]),  # Имя для модели
                    'residence_since': int(data[10]),  # Имя для БД
                    'property': int(data[11]),
                    'age': int(data[12]),
                    'other_installment_plans': int(data[13]),
                    'housing': int(data[14]),
                    'existing_credits': int(data[15]),
                    'job': int(data[16]),
                    'num_dependents': int(data[17]),
                    'telephone': int(data[18]),
                    'foreign_worker': int(data[19])
                }
            except (ValueError, IndexError) as e:
                await update.message.reply_text(f"Ошибка в формате данных: {str(e)}. Проверьте, что все поля содержат корректные числовые значения.")
                return
            
            # Получаем предсказание
            result = await self.predict_credit_risk(user_data)
            
            if "error" in result:
                await update.message.reply_text(f"Ошибка при предсказании: {result['error']}")
                return
            
            # Определяем риск-факторы
            risk_factors = []
            if user_data['status'] == 1:
                risk_factors.append("отрицательный баланс на счете")
            if user_data['duration'] > 24:
                risk_factors.append("длительный срок кредита")
            if user_data['credit_history'] >= 3:
                risk_factors.append("проблемы с кредитной историей")
            if user_data['credit_amount'] > 10000:
                risk_factors.append("большая сумма кредита")
            if user_data['savings'] <= 1:
                risk_factors.append("малые или отсутствующие сбережения")
            if user_data['employment'] <= 2:
                risk_factors.append("безработный или небольшой стаж работы")
            if user_data['installment_rate'] == 1:
                risk_factors.append("высокие ежемесячные выплаты")
            if user_data['age'] < 25:
                risk_factors.append("молодой возраст")
            
            # Добавляем описания для более понятного вывода
            for field in ['status', 'purpose', 'credit_history', 'savings']:
                if user_data[field] in FEATURE_MAPPINGS.get(field, {}):
                    user_data[f"{field}_desc"] = FEATURE_MAPPINGS[field][user_data[field]]
            
            # Для employment используем специальное маппирование
            if user_data['employment'] in FEATURE_MAPPINGS.get('employment', {}):
                user_data['employment_desc'] = FEATURE_MAPPINGS['employment'][user_data['employment']]
            
            # Расчет ежемесячного платежа
            monthly_payment = user_data['credit_amount'] / user_data['duration']
            
            # Формируем сообщение
            prediction_text = "Кредит будет одобрен" if result["prediction"] == 1 else "Кредит не будет одобрен"
            result_message = f"Предсказание: {prediction_text}\n"
            result_message += f"Вероятность одобрения кредита: {result['probability']:.2f}\n"
            
            # Показываем примененный порог вместо базового
            result_message += f"Применённый порог одобрения: {result['applied_threshold']:.2f}\n"
            
            # Особое предупреждение при высоком риске и положительном предсказании
            if result['risk_score'] >= 8 and result["prediction"] == 1:
                result_message += f"⚠️ ВНИМАНИЕ! Высокий фактор риска: {result['risk_score']} из 10\n"
                result_message += "Рекомендуется дополнительная проверка клиента.\n\n"
            else:
                result_message += f"Фактор риска: {result['risk_score']} из 10\n\n"
            
            if risk_factors:
                result_message += "Выявленные факторы риска:\n"
                for factor in risk_factors:
                    result_message += f"- {factor}\n"
                result_message += "\n"
            
            result_message += "Основные параметры кредита:\n"
            result_message += f"Сумма кредита: {user_data['credit_amount']}\n"
            result_message += f"Срок кредита: {user_data['duration']} месяцев\n"
            result_message += f"Ежемесячный платеж: {monthly_payment:.2f}\n"
            result_message += f"Цель кредита: {user_data.get('purpose_desc', user_data['purpose'])}\n\n"
            
            result_message += "Данные клиента:\n"
            result_message += f"Статус счета: {user_data.get('status_desc', user_data['status'])}\n"
            result_message += f"Кредитная история: {user_data.get('credit_history_desc', user_data['credit_history'])}\n"
            result_message += f"Сбережения: {user_data.get('savings_desc', user_data['savings'])}\n"
            result_message += f"Стаж работы: {user_data.get('employment_desc', user_data['employment'])}\n"
            result_message += f"Возраст: {user_data['age']}\n"
                
            await update.message.reply_text(result_message)
            
            # Сохраняем данные в базу данных
            db_save_success = await self.save_data_to_db(
                user_data, 
                result['prediction_label'], 
                result['probability'], 
                result['risk_score']
            )
            
            if not db_save_success:
                await update.message.reply_text("Примечание: данные не были сохранены в базе данных из-за ошибки.")
        
        except Exception as e:
            logger.error(f"Ошибка в обработке данных: {e}")
            await update.message.reply_text(f"Произошла ошибка при обработке данных: {str(e)}")

    async def importance_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /importance для просмотра важности признаков"""
        try:
            if not hasattr(self.model, 'feature_importances_'):
                await update.message.reply_text("Эта модель не поддерживает оценку важности признаков.")
                return
                
            # Получаем важность признаков
            importances = self.model.feature_importances_
            
            # Определяем имена признаков
            if not self.feature_names:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            else:
                # Обрезаем список имен до длины importances
                feature_names = self.feature_names[:len(importances)]
            
            # Создаем DataFrame с важностью признаков
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Формируем сообщение
            message = f"Важность признаков для принятия решения (модель {MODEL_VERSION}):\n\n"
            
            # Выводим топ-15 признаков или все, если их меньше
            top_n = min(15, len(feature_importance_df))
            for i, (feature, importance) in enumerate(
                zip(feature_importance_df['Feature'].head(top_n), feature_importance_df['Importance'].head(top_n))
            ):
                message += f"{i+1}. {feature}: {importance:.4f}\n"
                
            message += f"\nОптимальный порог для принятия решения: {self.optimal_threshold:.3f}"
            
            await update.message.reply_text(message)
        except Exception as e:
            await update.message.reply_text(f"Ошибка при получении важности признаков: {e}")

    async def stats_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats для просмотра статистики"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Получаем общую статистику
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN prediction = 'Одобрен' THEN 1 ELSE 0 END) as approved,
                        SUM(CASE WHEN prediction = 'Отклонен' THEN 1 ELSE 0 END) as rejected,
                        AVG(probability) as avg_probability,
                        AVG(risk_score) as avg_risk
                    FROM credit_scoring_predictions
                """)
                result = cursor.fetchone()
                
                if not result or result[0] == 0:
                    await update.message.reply_text("Нет данных для статистики.")
                    return
                    
                total, approved, rejected, avg_prob, avg_risk = result
                
                # Формируем сообщение
                message = f"Статистика предсказаний (модель {MODEL_VERSION}):\n"
                message += f"Всего запросов: {total}\n"
                message += f"Одобрено: {approved} ({approved/total*100:.1f}%)\n"
                message += f"Отклонено: {rejected} ({rejected/total*100:.1f}%)\n"
                message += f"Средняя вероятность одобрения: {avg_prob:.2f}\n"
                message += f"Средний фактор риска: {avg_risk:.2f} из 10\n"
                
                # Добавляем статистику по суммам кредитов
                cursor.execute("""
                    SELECT 
                        MIN(amount) as min_amount,
                        MAX(amount) as max_amount,
                        AVG(amount) as avg_amount,
                        AVG(CASE WHEN prediction = 'Одобрен' THEN amount ELSE NULL END) as avg_approved,
                        AVG(CASE WHEN prediction = 'Отклонен' THEN amount ELSE NULL END) as avg_rejected
                    FROM credit_scoring_predictions
                """)
                amount_result = cursor.fetchone()
                
                if amount_result:
                    min_amount, max_amount, avg_amount, avg_approved, avg_rejected = amount_result
                    message += f"\nСтатистика по суммам кредитов:\n"
                    message += f"Минимальная сумма: {min_amount}\n"
                    message += f"Максимальная сумма: {max_amount}\n"
                    message += f"Средняя сумма: {avg_amount:.2f}\n"
                    message += f"Средняя сумма одобренных: {avg_approved:.2f if avg_approved else 0}\n"
                    message += f"Средняя сумма отклоненных: {avg_rejected:.2f if avg_rejected else 0}\n"
                
                await update.message.reply_text(message)
                
        except Exception as e:
            await update.message.reply_text(f"Ошибка при получении статистики: {e}")

    async def compare_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /compare для получения примеров клиентов"""
        try:
            args = context.args
            
            if not args:
                await update.message.reply_text("Укажите тип клиента для получения примера: /compare good, /compare bad или /compare average")
                return
            
            client_type = args[0].lower()
            
            if client_type in ["good", "хороший"]:
                client = CLIENT_EXAMPLES["good"]
                client_desc = "хорошего"
            elif client_type in ["bad", "плохой"]:
                client = CLIENT_EXAMPLES["bad"]
                client_desc = "плохого"
            elif client_type in ["average", "средний"]:
                client = CLIENT_EXAMPLES["average"]
                client_desc = "среднего"
            else:
                await update.message.reply_text("Неизвестный тип клиента. Используйте: /compare good, /compare bad или /compare average")
                return
                
            # Преобразуем командную строку для использования
            command_line = "/predict "
            command_line += ", ".join(str(value) for value in client.values())
            
            message = f"Пример {client_desc} клиента. Скопируйте и отправьте следующую команду:\n\n{command_line}"
            await update.message.reply_text(message)
            
        except Exception as e:
            await update.message.reply_text(f"Ошибка при формировании примера: {e}")

    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = (
            "Справка по использованию бота кредитного скоринга:\n\n"
            "Основные команды:\n"
            "/start - Начало работы с ботом\n"
            "/predict - Оценка кредитоспособности по указанным параметрам\n"
            "/stats - Получение статистики по предсказаниям\n"
            "/importance - Просмотр важности признаков для модели\n"
            "/compare - Примеры клиентов (good, bad, average)\n"
            "/help - Эта справка\n\n"
            
            "Формат данных для команды /predict:\n"
            "status, duration, credit_history, purpose, amount, savings, employment_duration, "
            "installment_rate, personal_status_sex, other_debtors, residence_since, property, "
            "age, other_installment_plans, housing, existing_credits, job, num_dependents, "
            "telephone, foreign_worker\n\n"
            
            "Описание основных параметров:\n"
            "status - статус чекового счета (1-4)\n"
            "duration - срок кредита в месяцах\n"
            "credit_history - кредитная история (0-4)\n"
            "purpose - цель кредита (0-10)\n"
            "amount - сумма кредита\n"
            "savings - уровень сбережений (1-5)\n"
            "employment_duration - стаж работы (1-5)\n"
            
            "Пример вызова команды:\n"
            "/predict 4, 12, 2, 0, 4000, 4, 5, 4, 1, 1, 4, 1, 40, 1, 1, 1, 4, 1, 1, 1"
        )
        await update.message.reply_text(help_text)

def main():
    """Основная функция для запуска бота"""
    # Создаем бот
    bot = CreditScoringBot()
    
    # Токен, полученный от BotFather
    token = '7880641469:AAH7VjvljG0Pv4oS7r7_VPD41sfsg7cTaoY'
    
    # Создаем приложение
    app = Application.builder().token(token).build()
    
    # Добавляем обработчики команд
    app.add_handler(CommandHandler("start", bot.start_handler))
    app.add_handler(CommandHandler("predict", bot.predict_handler))
    app.add_handler(CommandHandler("stats", bot.stats_handler))
    app.add_handler(CommandHandler("importance", bot.importance_handler))
    app.add_handler(CommandHandler("compare", bot.compare_handler))
    app.add_handler(CommandHandler("help", bot.help_handler))
    
    # Запускаем приложение
    logger.info(f"Запуск бота с моделью {MODEL_VERSION}...")
    app.run_polling(poll_interval=1.0)

if __name__ == '__main__':
    main()