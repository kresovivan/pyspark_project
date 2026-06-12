"""
PDF → CSV: Английские слова с IPA, русской транскрипцией, переводом и частями речи
Переписанная версия с более адекватным определением частей речи.
"""

import re
import pdfplumber
import csv
import time
import os
import json
from collections import Counter
from pathlib import Path

# --- Лингвистические библиотеки ---
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords, wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ nltk не установлен. Установите: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️ Модель spaCy не найдена. Установите: python -m spacy download en_core_web_sm")
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("⚠️ spaCy не установлен. Установите: pip install spacy")

try:
    from eng_to_ipa import convert as ipa_convert
    ENG_TO_IPA_AVAILABLE = True
except ImportError:
    ENG_TO_IPA_AVAILABLE = False
    print("⚠️ eng_to_ipa не установлен. Установите: pip install eng-to-ipa")

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ deep_translator не установлен. Установите: pip install deep-translator")


# ============================================================
# СЛОВАРЬ ЧАСТЕЙ РЕЧИ: английский → русский
# ============================================================

POS_EN_TO_RU = {
    'noun': 'существительное',
    'verb': 'глагол',
    'adj': 'прилагательное',
    'adv': 'наречие',
    'ADJ': 'прилагательное',
    'ADP': 'предлог/послелог',
    'ADV': 'наречие',
    'AUX': 'вспомогательный глагол',
    'CONJ': 'союз',
    'CCONJ': 'сочинительный союз',
    'DET': 'артикль/определитель',
    'INTJ': 'междометие',
    'NOUN': 'существительное',
    'NUM': 'числительное',
    'PART': 'частица',
    'PRON': 'местоимение',
    'PROPN': 'имя собственное',
    'PUNCT': 'пунктуация',
    'SCONJ': 'подчинительный союз',
    'SYM': 'символ',
    'VERB': 'глагол',
    'X': 'прочее',
    'SPACE': 'пробел',
}

# --- Конфигурация ---
CONFIG = {
    'min_word_length': 3,
    'max_word_length': 25,
    'batch_size': 100,
    'sleep_between_requests': 0.3,
    'use_lemmatization': True,
    'filter_stopwords': True,
    'filter_technical': True,

    # >>> РЕЖИМ определения части речи:
    # 'wordnet'  – словарный POS по лемме (рекомендуется)
    # 'spacy'    – контекстный POS, использовать если будешь добавлять контекст
    # None       – не определять POS
    'pos_mode': 'wordnet',
}

# --- Пути (замени под себя) ---
PDF_PATH = r"C:\Users\kresovivan\Programming\pyspark_project\mastering-object-oriented-programming-with-python.pdf"
CSV_PATH = r"C:\Users\kresovivan\Programming\pyspark_project\output_full.csv"
CACHE_PATH = r"C:\Users\kresovivan\Programming\pyspark_project\translation_cache.json"


# --- Технические артефакты ---
TECH_PATTERNS = [
    r'^[a-z]+_[a-z_]+$',
    r'^[a-z]+[A-Z][a-zA-Z]*$',
    r'^[a-z]+\.[a-z]+$',
    r'^[a-z]+::[a-z]+$',
    r'^pyspark.*$',
    r'^block_\d+$',
    r'^[a-z]+[0-9]+$',
    r'^[0-9]+[a-z]+$',
    r'^[a-f0-9]{8,}$',
    r'^[a-z]+://.*$',
]

TECH_REGEX = [re.compile(p) for p in TECH_PATTERNS]


# ============================================================
# БЛОК 1: ИНИЦИАЛИЗАЦИЯ
# ============================================================

def init_nltk():
    """Скачивает необходимые данные NLTK."""
    if not NLTK_AVAILABLE:
        return None, set()

    required = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/omw-1.4', 'omw-1.4'),
    ]

    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"   📥 Скачиваю NLTK: {name}...")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"   ⚠️ Не удалось скачать {name}: {e}")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tech_stopwords = {
        'pyspark', 'spark', 'dataframe', 'rdd', 'scala', 'python',
        'hdfs', 'yarn', 'json', 'csv', 'sql', 'api', 'url', 'http',
        'www', 'com', 'org', 'github', 'stackoverflow', 'udf',
        'jdbc', 'hive', 'kafka', 'avro', 'parquet', 'orc',
    }
    stop_words.update(tech_stopwords)

    return lemmatizer, stop_words


def normalize_word(word: str, lemmatizer=None) -> str:
    """Нормализует слово: lower + strip + лемматизация."""
    word = word.lower().strip()
    if lemmatizer and CONFIG['use_lemmatization']:
        word = lemmatizer.lemmatize(word)
    return word


# ============================================================
# БЛОК 2: ЧАСТИ РЕЧИ — СЛОВАРНЫЙ И КОНТЕКСТНЫЙ ПОДХОД
# ============================================================

def wordnet_pos_from_synsets(lemma: str) -> str:
    """
    Определяет часть речи через WordNet по лемме.
    Возвращает одно из: noun, verb, adj, adv или ''.
    """
    if not NLTK_AVAILABLE:
        return ''

    lemma = lemma.lower()
    synsets = wordnet.synsets(lemma)
    if not synsets:
        return ''

    # Берем самый частый синсет и его POS
    pos = synsets[0].pos()
    mapping = {
        'n': 'noun',
        'v': 'verb',
        'a': 'adj',
        's': 'adj',
        'r': 'adv',
    }
    return mapping.get(pos, '')


def spacy_pos_single_word(word: str) -> str:
    """
    Контекстный POS через spaCy на одном слове.
    Использовать осторожно, лучше на предложениях.
    """
    if not SPACY_AVAILABLE or nlp is None:
        return ''
    doc = nlp(word)
    if not doc:
        return ''
    return doc[0].pos_  # NOUN, VERB, ADJ, ADV, ...


def get_pos_labels(word: str, lemma: str) -> tuple[str, str, str]:
    """
    Возвращает три вещи:
    - pos_mode: 'wordnet' / 'spacy' / 'none'
    - pos_en: 'noun' / 'verb' / 'adj' / 'adv' / 'NOUN' / ... / ''
    - pos_ru: русская подпись или ''
    """
    mode = CONFIG.get('pos_mode')

    if mode == 'wordnet':
        pos_en = wordnet_pos_from_synsets(lemma or word)
        pos_ru = POS_EN_TO_RU.get(pos_en, '')
        return 'wordnet', pos_en, pos_ru

    if mode == 'spacy':
        pos_en = spacy_pos_single_word(word)
        pos_ru = POS_EN_TO_RU.get(pos_en, '')
        return 'spacy', pos_en, pos_ru

    return 'none', '', ''


# ============================================================
# БЛОК 3: ТРАНСКРИПЦИЯ
# ============================================================

def ipa_to_russian(ipa_str: str) -> str:
    """Преобразует IPA-строку в русскую фонетическую транскрипцию."""
    if not ipa_str or '*' in ipa_str:
        return ""

    ipa_str = ipa_str.replace("'", "ˈ").replace("ˌ", "")

    mapping = {
        'eɪ': 'эй', 'aɪ': 'ай', 'ɔɪ': 'ой', 'aʊ': 'ау', 'əʊ': 'оу',
        'ɪə': 'иэ', 'eə': 'эа', 'ʊə': 'уа', 'juː': 'ю', 'jʊ': 'ю',
        'iː': 'и', 'uː': 'у', 'ɑː': 'а', 'ɔː': 'о', 'ɜː': 'ё', 'eː': 'э',
        'ɪ': 'и', 'ʊ': 'у', 'ʌ': 'а', 'æ': 'э', 'ɑ': 'а', 'ɔ': 'о',
        'ɛ': 'э', 'ə': 'э', 'e': 'э', 'ɒ': 'о', 'ɨ': 'ы', 'i': 'и',
        'p': 'п', 'b': 'б', 't': 'т', 'd': 'д', 'k': 'к', 'g': 'г',
        'f': 'ф', 'v': 'в', 'θ': 'с', 'ð': 'з', 's': 'с', 'z': 'з',
        'ʃ': 'ш', 'ʒ': 'ж', 'h': 'х', 'm': 'м', 'n': 'н', 'ŋ': 'н',
        'l': 'л', 'r': 'р', 'ɹ': 'р', 'w': 'у', 'j': 'й',
        'ʤ': 'дж', 'ʧ': 'ч', 'tʃ': 'ч', 'dʒ': 'дж',
        'ɚ': 'эр', 'əl': 'л', 'ən': 'н', 'əm': 'м',
        'ː': '', 'ˈ': '', 'ˌ': '', '‿': '',
    }

    for code, rus in sorted(mapping.items(), key=lambda x: -len(x[0])):
        ipa_str = ipa_str.replace(code, rus)

    ipa_str = re.sub(r'\s+', ' ', ipa_str).strip()

    lat_to_cyr = {
        'i': 'и', 'u': 'у', 'o': 'о', 'e': 'е', 'a': 'а',
        'c': 'к', 'b': 'б', 'h': 'х', 'y': 'ы', 'p': 'п',
        's': 'с', 't': 'т', 'k': 'к', 'd': 'д', 'f': 'ф',
        'g': 'г', 'j': 'дж', 'l': 'л', 'm': 'м', 'n': 'н',
        'r': 'р', 'v': 'в', 'w': 'у', 'z': 'з'
    }
    for lat, cyr in lat_to_cyr.items():
        ipa_str = ipa_str.replace(lat, cyr)

    ipa_str = re.sub(r'\s+', ' ', ipa_str).strip()
    return ipa_str


def get_ipa(word: str) -> str:
    """Получает IPA-транскрипцию."""
    if not ENG_TO_IPA_AVAILABLE:
        return ""
    try:
        ipa = ipa_convert(word)
        return ipa if ipa and '*' not in ipa else ""
    except Exception:
        return ""


# ============================================================
# БЛОК 4: ПЕРЕВОД С КЭШИРОВАНИЕМ
# ============================================================

class TranslationCache:
    """Кэш переводов в JSON."""

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                print(f"📂 Загружен кэш переводов: {len(self.cache)} записей")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки кэша: {e}")
                self.cache = {}

    def save(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кэша: {e}")

    def get(self, word: str) -> str | None:
        word = word.lower()
        if word in self.cache:
            self.hits += 1
            return self.cache[word]
        self.misses += 1
        return None

    def set(self, word: str, translation: str):
        self.cache[word.lower()] = translation

    def stats(self) -> str:
        total = self.hits + self.misses
        if total == 0:
            return "Кэш не использовался"
        rate = self.hits / total * 100
        return f"Кэш: {self.hits} попаданий, {self.misses} промахов ({rate:.1f}%)"


def translate_word(word: str, translator, cache: TranslationCache) -> str:
    """Переводит слово с использованием кэша."""
    cached = cache.get(word)
    if cached is not None:
        return cached

    if not TRANSLATOR_AVAILABLE or translator is None:
        return ""

    try:
        translation = translator.translate(word)
        cache.set(word, translation)
        time.sleep(CONFIG['sleep_between_requests'])
        return translation
    except Exception as e:
        print(f"   ⚠️ Ошибка перевода '{word}': {e}")
        return ""


# ============================================================
# БЛОК 5: ИЗВЛЕЧЕНИЕ И ФИЛЬТРАЦИЯ СЛОВ
# ============================================================

def is_technical_artifact(word: str) -> bool:
    for pattern in TECH_REGEX:
        if pattern.match(word):
            return True
    return False


def clean_pdf_text(text: str) -> str:
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()


def extract_words_from_pdf(pdf_path: str, lemmatizer=None, stop_words=None) -> list[str]:
    print("📖 Открываю PDF...")
    all_words = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Всего страниц: {total_pages}")

        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if not text:
                continue

            cleaned = clean_pdf_text(text)
            words = re.findall(r'\b[a-z]{3,}\b', cleaned)
            all_words.extend(words)

            if i % 50 == 0 or i == total_pages:
                print(f"   Обработано {i}/{total_pages} страниц... ({len(all_words)} слов)")

    print(f"\n📊 Извлечено {len(all_words):,} слов (сырых)")

    if CONFIG['use_lemmatization'] and lemmatizer:
        print("🔤 Лемматизация слов...")
        all_words = [lemmatizer.lemmatize(w) for w in all_words]

    filtered = []
    skipped_stats = Counter()

    for word in all_words:
        if not (CONFIG['min_word_length'] <= len(word) <= CONFIG['max_word_length']):
            skipped_stats['length'] += 1
            continue

        if CONFIG['filter_stopwords'] and stop_words and word in stop_words:
            skipped_stats['stopword'] += 1
            continue

        if CONFIG['filter_technical'] and is_technical_artifact(word):
            skipped_stats['technical'] += 1
            continue

        filtered.append(word)

    print(f"📊 После фильтрации: {len(filtered):,} слов")
    if skipped_stats:
        print(f"   Пропущено: {dict(skipped_stats)}")

    return filtered


# ============================================================
# БЛОК 6: РАБОТА С CSV
# ============================================================

def load_existing_csv(csv_path: str, lemmatizer=None) -> set[str]:
    existing = set()

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return existing

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'English Term' in row and row['English Term']:
                    word = normalize_word(row['English Term'], lemmatizer)
                    existing.add(word)

        if existing:
            print(f"📂 Загружено {len(existing)} слов из CSV (после нормализации)")
        else:
            print("📂 CSV-файл пуст, будет создан новый.")
    except Exception as e:
        print(f"⚠️ Ошибка чтения CSV: {e}")

    return existing


def save_batch(batch: list[dict], csv_path: str, mode: str):
    if not batch:
        return

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    headers = [
        'English Term', 'Lemma',
        'IPA', 'Russian Pronunciation',
        'Russian Translation',
        'POS Mode',
        'Part of Speech (EN)', 'Part of Speech (RU)',
        'Frequency'
    ]

    if mode == 'w' or not file_exists:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for item in batch:
                writer.writerow([
                    item['word'], item['lemma'],
                    item['ipa'], item['rus_pron'],
                    item['translation'],
                    item['pos_mode'],
                    item['pos_en'], item['pos_ru'],
                    item.get('frequency', 1)
                ])
    else:
        with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for item in batch:
                writer.writerow([
                    item['word'], item['lemma'],
                    item['ipa'], item['rus_pron'],
                    item['translation'],
                    item['pos_mode'],
                    item['pos_en'], item['pos_ru'],
                    item.get('frequency', 1)
                ])


# ============================================================
# БЛОК 7: ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("=" * 70)
    print("📚 PDF → CSV: Словарь с IPA, транскрипцией и частями речи (новая версия)")
    print("=" * 70)

    lemmatizer, stop_words = init_nltk() if NLTK_AVAILABLE else (None, set())
    cache = TranslationCache(CACHE_PATH)
    translator = GoogleTranslator(source='en', target='ru') if TRANSLATOR_AVAILABLE else None

    # 1. Извлечение слов
    all_words = extract_words_from_pdf(PDF_PATH, lemmatizer, stop_words)
    if not all_words:
        print("❌ Не удалось извлечь слова из PDF")
        return

    word_freq = Counter(all_words)
    unique_words = sorted(word_freq.keys())

    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Всего слов (с повторами): {len(all_words):,}")
    print(f"   Уникальных слов:          {len(unique_words):,}")
    print(f"   Средняя частотность:      {len(all_words)/len(unique_words):.1f}")

    # 2. Загрузка существующего CSV
    existing = load_existing_csv(CSV_PATH, lemmatizer)

    if existing:
        new_words = [w for w in unique_words if w not in existing]

        print(f"\n📂 В CSV уже {len(existing)} слов (после нормализации)")
        print(f"🆕 Новых для обработки: {len(new_words)}")

        if not new_words:
            print("✅ Все слова уже в CSV!")
            return

        choice = input("\nДействие: [a] добавить новые, [r] перезаписать всё, [q] выход: ").lower()
        if choice == 'q':
            return
        elif choice == 'r':
            words_to_process = unique_words
            mode = 'w'
        else:
            words_to_process = new_words
            mode = 'a'
    else:
        words_to_process = unique_words
        mode = 'w'
        print("\n🆕 Создаём новый CSV")

    if not words_to_process:
        print("✅ Нечего обрабатывать")
        return

    print(f"\n🔄 Обработка {len(words_to_process)} слов...")
    print(f"   Режим POS: {CONFIG['pos_mode']}")
    print(f"   Задержка между переводами: {CONFIG['sleep_between_requests']}с")

    batch = []
    saved_total = 0
    skipped = 0
    current_mode = mode

    try:
        for idx, word in enumerate(words_to_process, 1):
            lemma = normalize_word(word, lemmatizer)

            ipa = get_ipa(lemma)
            if not ipa:
                skipped += 1
                continue

            rus_pron = ipa_to_russian(ipa)
            if not rus_pron:
                skipped += 1
                continue

            translation = translate_word(lemma, translator, cache)

            pos_mode, pos_en, pos_ru = get_pos_labels(word, lemma)

            batch.append({
                'word': word,
                'lemma': lemma,
                'ipa': ipa,
                'rus_pron': rus_pron,
                'translation': translation,
                'pos_mode': pos_mode,
                'pos_en': pos_en,
                'pos_ru': pos_ru,
                'frequency': word_freq[word],
            })

            if idx % 10 == 0 or idx == len(words_to_process):
                print(f"   {idx}/{len(words_to_process)}: {word} ({lemma}) "
                      f"| {ipa} | {rus_pron} | POS={pos_en or '??'} | {translation[:25] if translation else '...'}")

            if len(batch) >= CONFIG['batch_size']:
                save_batch(batch, CSV_PATH, current_mode)
                saved_total += len(batch)
                print(f"      💾 Сохранено {saved_total} (пропущено {skipped})")
                batch = []
                if current_mode == 'w':
                    current_mode = 'a'

        if batch:
            save_batch(batch, CSV_PATH, current_mode)
            saved_total += len(batch)

        cache.save()

        print(f"\n{'='*70}")
        print(f"✅ ГОТОВО!")
        print(f"   Сохранено слов: {saved_total}")
        print(f"   Пропущено (нет IPA/транскрипции): {skipped}")
        print(f"   {cache.stats()}")
        print(f"   CSV: {CSV_PATH}")
        print(f"   Кэш: {CACHE_PATH}")
        print(f"{'='*70}")

    except KeyboardInterrupt:
        print("\n\n⛔ Прервано пользователем")
        if batch:
            save_batch(batch, CSV_PATH, current_mode)
            print(f"💾 Сохранено {len(batch)} слов (частичный результат)")
        cache.save()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        if batch:
            save_batch(batch, CSV_PATH, current_mode)
        cache.save()
        raise


if __name__ == "__main__":
    main()