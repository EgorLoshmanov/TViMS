import argparse     # библиотека для чтения аргументов из командной строки
import re            # регулярные выражения — используются для очистки текста
import pandas as pd  # библиотека для работы с таблицами (DataFrame)
import math          # математика (используется для логарифмов)
import json          # сохранение и загрузка обученной модели в JSON-файл

# === ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
def build_data_file(path_to_csv):
    """
    Загружает CSV-файл с датасетом, оставляет нужные столбцы и очищает текст.
    Возвращает готовый DataFrame с колонками 'text' и 'label'.
    """
    # Чтение CSV и выбор только нужных столбцов
    df = pd.read_csv(path_to_csv)
    df = df[['Message', 'Spam/Ham']]
    df.columns = ['text', 'label']

    # Перемешиваем строки, чтобы данные не шли подряд по метке
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Функция для очистки текста: оставляем только буквы, приводим к нижнему регистру
    def clean_text(text):
        text = str(text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # удаляем всё кроме букв и пробелов
        text = text.lower()
        text = ' '.join(text.split())             # убираем лишние пробелы
        return list(set(text.split()))            # разбиваем на уникальные слова

    # Применяем очистку к каждому сообщению
    df['text'] = df['text'].apply(clean_text)
    return df

# === ПРОВЕРКА МЕТКИ ===
def is_spam(df, idx):
    """Возвращает True, если строка с индексом idx помечена как spam."""
    return df.loc[idx, "label"] == 'spam'

# === КЛАСС НАИВНОГО БАЙЕСОВСКОГО КЛАССИФИКАТОРА ===
class NBC:
    """
    NBC — реализация Наивного Байесовского классификатора (Naive Bayes Classifier)
    для фильтрации спама.
    """
    def __init__(self):
        # Вероятности слова для каждого класса
        self.word_probs = {}   # вероятности слова для 'spam' и 'ham'
        self.word_freq = {}    # количество вхождений слова в спам и не-спам
        # Априорные вероятности (P(spam) и P(ham))
        self.priors_probs = {"spam": 0.5, "ham": 0.5}
        self.priors_freq = {"spam": 0, "ham": 0}
        self.count_letters = 0  # количество обработанных писем

    # --- Обучение модели ---
    def init(self, dataset, start_idx, finish_idx):
        """
        Обучает модель на части датасета (от start_idx до finish_idx).
        Считает частоты слов и вероятности для Naive Bayes.
        """
        for i in range(start_idx, finish_idx):
            text = dataset.loc[i, "text"]
            label = dataset.loc[i, "label"]

            # пропускаем пустые строки
            if text == ['nan']:
                continue

            # проходим по каждому слову из письма
            for word in text:
                # если слово новое — создаём записи в словарях
                if word not in self.word_freq:
                    self.word_probs[word] = {'spam': 0, 'ham': 0}
                    self.word_freq[word] = {'spam': 0, 'ham': 0}

                # увеличиваем счётчик для нужного класса
                if is_spam(dataset, i):
                    self.word_freq[word]['spam'] += 1
                else:
                    self.word_freq[word]['ham'] += 1

                # считаем общие частоты и вероятности с Лапласовским сглаживанием
                word_freq_spam = self.word_freq[word]['spam']
                word_freq_ham = self.word_freq[word]['ham']
                total_spam = sum(freqs['spam'] for freqs in self.word_freq.values())
                total_ham = sum(freqs['ham'] for freqs in self.word_freq.values())

                self.word_probs[word]['spam'] = (word_freq_spam + 1) / (total_spam + 2)
                self.word_probs[word]['ham'] = (word_freq_ham + 1) / (total_ham + 2)

            # обновляем априорные вероятности
            self.count_letters += 1
            if is_spam(dataset, i):
                self.priors_freq['spam'] += 1
            else:
                self.priors_freq['ham'] += 1

            self.priors_probs['spam'] = self.priors_freq['spam'] / self.count_letters
            self.priors_probs['ham'] = self.priors_freq['ham'] / self.count_letters

            print(f"{i} email with label '{label}' was analyzed")

    # --- Классификация ---
    def is_spam(self, text):
        """
        Классифицирует письмо как спам или не спам (0/1) по логарифмам вероятностей.
        """
        if not text:
            return 0

        # используем логарифмы, чтобы избежать переполнения при маленьких вероятностях
        log_spam = math.log(self.priors_probs['spam'])
        log_ham = math.log(self.priors_probs['ham'])

        # добавляем вероятности для каждого слова
        for word in text:
            if word in self.word_probs:
                log_spam += math.log(self.word_probs[word]['spam'])
                log_ham += math.log(self.word_probs[word]['ham'])

        # возвращаем 1, если вероятность спама выше
        return 1 if log_spam > log_ham else 0

    # --- Оценка качества модели ---
    def evaluate_model(self, dataset, start_idx, finish_idx):
        """
        Тестирует модель на отложенной выборке и выводит метрики:
        - accuracy (точность)
        - sensitivity (чувствительность, True Positive Rate)
        - specificity (специфичность, True Negative Rate)
        """
        tp = fp = tn = fn = 0

        for i in range(start_idx, finish_idx):
            text = dataset.loc[i, 'text']
            predicted_spam = self.is_spam(text)
            actual_spam = is_spam(dataset, i)

            # подсчёт TP, FP, TN, FN
            if predicted_spam:
                if actual_spam:
                    tp += 1
                else:
                    fp += 1
            else:
                if actual_spam:
                    fn += 1
                else:
                    tn += 1

            # подсчёт метрик по мере итерации
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total else 0
            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0

            print(f"{i}) predict: {'spam' if predicted_spam else 'ham'}; "
                  f"actual: {'spam' if actual_spam else 'ham'}")
            print(f"test: {i}, accuracy: {accuracy:.4f}, "
                  f"sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}")

        # итоговые метрики
        print("\n")
        print('-' * 35)
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        print(f"Точность: {accuracy:.4f}")
        print(f"Чувствительность: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

    # --- Сохранение/загрузка модели ---
    def save_model(self, name):
        """Сохраняет обученную модель в JSON-файл."""
        model_data = {
            'word_probs': self.word_probs,
            'priors_probs': self.priors_probs,
        }

        with open(f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"Model saved as {name}.json")

    def load_model(self, path):
        """Загружает сохранённую модель из JSON-файла."""
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        self.word_probs = model_data['word_probs']
        self.priors_probs = model_data['priors_probs']
        print(f"Model loaded from {path}")

# === ОСНОВНАЯ ФУНКЦИЯ ПРОГРАММЫ ===
def main():
    """
    Основная точка входа:
    - считывает путь к CSV через аргументы командной строки;
    - подготавливает данные;
    - загружает или обучает модель;
    - оценивает её на тестовой выборке.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")  # путь к CSV
    args = parser.parse_args()

    # загружаем и подготавливаем данные
    df = build_data_file(args.input_path)

    # инициализируем классификатор
    nbc = NBC()

    # Если модель уже сохранена:
    nbc.load_model('model_1.json')

    # Проверяем качество модели на части выборки
    nbc.evaluate_model(df, 10000, 13000)

if __name__ == '__main__':
    main()
