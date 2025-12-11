import argparse
import re
import pandas as pd
import math
import json
from typing import List, Tuple


def build_data_file(path_to_csv: str) -> pd.DataFrame:
    """Загружает CSV и возвращает перемешанный DataFrame с колонками: text(list[str]), label('spam'|'ham')."""
    # Попытка корректно прочитать файл с разными кодировками и разделителями
    try:
        df = pd.read_csv(path_to_csv)
    except UnicodeDecodeError:
        df = pd.read_csv(path_to_csv, encoding="latin1")

    # Проверяем, что есть нужные колонки 'Message' и 'Spam/Ham'
    if not {"Message", "Spam/Ham"}.issubset(df.columns):
        raise KeyError(
            "CSV должен содержать колонки 'Message' и 'Spam/Ham' (значения: 'spam'|'ham')."
        )

    df = df[["Message", "Spam/Ham"]].copy()
    df.columns = ["text", "label"]

    # Перемешиваем датасет для обучения
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    def clean_text(text: str) -> List[str]:
        text = str(text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        text = " ".join(text.split())
        words = list(set(text.split()))  # убираем дубликаты в сообщении
        return words

    df["text"] = df["text"].apply(clean_text)
    return df


def is_spam(df: pd.DataFrame, idx: int) -> bool:
    return df.loc[idx, "label"] == "spam"


class NBC:
    def __init__(self):
        self.word_probs: dict[str, dict[str, float]] = {}
        self.word_freq: dict[str, dict[str, int]] = {}
        self.priors_probs: dict[str, float] = {"spam": 0.5, "ham": 0.5}
        self.priors_freq: dict[str, int] = {"spam": 0, "ham": 0}

    # ===== очистка текста для предсказания =====
    def clean_text_for_prediction(self, text: str) -> List[str]:
        text = str(text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = text.lower()
        text = " ".join(text.split())
        words = list(set(text.split()))
        return words

    # ===== обучение =====
    def train(self, dataset: pd.DataFrame, start_idx: int, finish_idx: int) -> None:
        start_idx = max(0, start_idx)
        finish_idx = min(len(dataset), finish_idx)
        if start_idx >= finish_idx:
            raise ValueError("Пустой диапазон для обучения: проверь start/finish индексы.")

        # 1) собираем частоты слов
        for i in range(start_idx, finish_idx):
            text = dataset.loc[i, "text"]
            label = dataset.loc[i, "label"]
            if not text or text == ["nan"]:
                continue
            for word in text:
                if word not in self.word_freq:
                    self.word_freq[word] = {"spam": 0, "ham": 0}
                self.word_freq[word][label] += 1
            self.priors_freq[label] += 1

        # 2) вычисляем условные вероятности с добавлением Лапласа
        total_spam_words = sum(freqs["spam"] for freqs in self.word_freq.values())
        total_ham_words = sum(freqs["ham"] for freqs in self.word_freq.values())
        vocab_size = max(1, len(self.word_freq))

        self.word_probs = {}
        for word, freqs in self.word_freq.items():
            self.word_probs[word] = {
                "spam": (freqs["spam"] + 1) / (total_spam_words + vocab_size),
                "ham": (freqs["ham"] + 1) / (total_ham_words + vocab_size),
            }

        # 3) априорные вероятности классов
        total_examples = (finish_idx - start_idx)
        self.priors_probs = {
            "spam": self.priors_freq["spam"] / total_examples if total_examples else 0.5,
            "ham": self.priors_freq["ham"] / total_examples if total_examples else 0.5,
        }

    # ===== предсказание / оценка =====
    def _log_scores(self, words: List[str]) -> Tuple[float, float]:
        # защита от нулевых вероятностей
        spam_prior = self.priors_probs.get("spam", 0.5) or 1e-12
        ham_prior = self.priors_probs.get("ham", 0.5) or 1e-12
        log_spam = math.log(spam_prior)
        log_ham = math.log(ham_prior)
        for w in words:
            if w in self.word_probs:
                log_spam += math.log(self.word_probs[w]["spam"])  # type: ignore[index]
                log_ham += math.log(self.word_probs[w]["ham"])    # type: ignore[index]
        return log_spam, log_ham

    def calculate_probabilities(self, words: List[str]) -> Tuple[float, float]:
        if not words:
            return 0.5, 0.5
        log_spam, log_ham = self._log_scores(words)
        # переводим обратно в вероятности с численной устойчивостью
        m = max(log_spam, log_ham)
        spam_prob = math.exp(log_spam - m)
        ham_prob = math.exp(log_ham - m)
        total = spam_prob + ham_prob
        return spam_prob / total, ham_prob / total

    def is_spam(self, words: List[str]) -> int:
        if not words:
            return 0
        log_spam, log_ham = self._log_scores(words)
        return 1 if log_spam > log_ham else 0

    def check_custom_text(self, text: str) -> int:
        words = self.clean_text_for_prediction(text)
        pred = self.is_spam(words)
        spam_p, ham_p = self.calculate_probabilities(words)
        print(f"\n Текст: '{text}'")
        print(f"Результат: {'СПАМ' if pred else 'НЕ СПАМ'}")
        print(f"Вероятность спама: {spam_p:.4f}")
        print(f"Вероятность не спама: {ham_p:.4f}")
        return pred

    def evaluate_model(self, dataset: pd.DataFrame, start_idx: int, finish_idx: int) -> None:
        start_idx = max(0, start_idx)
        finish_idx = min(len(dataset), finish_idx)

        tp = fp = tn = fn = 0
        for i in range(start_idx, finish_idx):
            words = dataset.loc[i, "text"]
            predicted_spam = self.is_spam(words)
            actual_spam = is_spam(dataset, i)

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

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0

            pred_lbl = "spam" if predicted_spam else "ham"
            act_lbl = "spam" if actual_spam else "ham"
            print(f"{i}) predict: {pred_lbl}; actual: {act_lbl}")
            print(
                f"test: {i}, accuracy: {accuracy:.4f}, sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}"
            )

        print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
        print(f"Истинно положительные (TP): {tp}")
        print(f"Ложно положительные (FP): {fp}")
        print(f"Истинно отрицательные (TN): {tn}")
        print(f"Ложно отрицательные (FN): {fn}")
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        print(f"Точность (Accuracy): {accuracy:.4f}")
        print(f"Чувствительность (Sensitivity): {sensitivity:.4f}")
        print(f"Специфичность (Specificity): {specificity:.4f}")

    def save_model(self, name: str) -> None:
        model_data = {
            "word_probs": self.word_probs,
            "priors_probs": self.priors_probs,
        }
        with open(f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print(f"Модель сохранена в {name}.json")

    def load_model(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
        self.word_probs = model_data["word_probs"]
        self.priors_probs = model_data["priors_probs"]
        print(f"Модель загружена из {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Наивный Байес для классификации спама")
    parser.add_argument("input_path", help="Путь к CSV с колонками: Message, Spam/Ham")
    parser.add_argument("--train-size", type=int, default=1000, help="Сколько строк использовать для обучения при первом запуске")
    parser.add_argument("--model", default="model_1.json", help="Файл модели для загрузки/сохранения")
    args = parser.parse_args()

    df = build_data_file(args.input_path)
    nbc = NBC()

    # загрузка или обучение модели
    trained_now = False
    try:
        nbc.load_model(args.model)
        print("Модель загружена")
    except FileNotFoundError:
        print("Модель не найдена, начинается обучение...")
        train_n = min(args.train_size, len(df))
        nbc.train(df, 0, train_n)
        nbc.save_model(args.model.replace(".json", "").rstrip("."))
        print("Модель обучена и сохранена")
        trained_now = True

    # === ОЦЕНКА НА ОТЛОЖЕННОЙ ВЫБОРКЕ ===
    if trained_now:
        # тестируем на хвосте, не попавшем в обучение
        test_start = train_n
        test_end = min(train_n + 500, len(df))
    else:
        # модель загружена: просто возьмём последние 500 строк как тест
        test_end = len(df)
        test_start = max(0, test_end - 500)

    if test_start < test_end:
        print("\n=== ОЦЕНКА НА ОТЛОЖЕННОЙ ВЫБОРКЕ ===")
        nbc.evaluate_model(df, test_start, test_end)

    # интерактивный режим
    print("\n" + "=" * 50)
    print("РЕЖИМ ПРОВЕРКИ ТЕКСТА")
    print("=" * 50)
    while True:
        try:
            user_text = input("\nВведите текст для проверки (или 'exit' для выхода):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Выход")
            break
        if user_text.lower() in {"exit", "выход", "quit"}:
            break
        if not user_text:
            print("Пустая строка!")
            continue
        nbc.check_custom_text(user_text)


if __name__ == "__main__":
    main()