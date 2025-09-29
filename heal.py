import pandas as pd
import numpy as np
import re
import os
import torch
import lightgbm as lgb
import joblib
import shutil
from pathlib import Path
from tqdm.auto import tqdm

tqdm.pandas()

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from datasets import Dataset
from peft import PeftModel, PeftConfig

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Используемое устройство: {DEVICE}")

CACHE_DIR = Path('./cache_hyperdrive')
FINAL_MODEL_PATH = Path('./production_model_assembled')  # Имя финальной, собранной модели

LOCAL_DEOBFUSCATOR_PATH = Path('./local_deobfuscator')
LOCAL_CLASSIFIER_PATH = Path('./local_classifier')

N_SPLITS = 2
RANDOM_STATE = 42
NUM_ROWS_TO_USE = None

PROFANITY_WORDS = [
    'хуй', 'хуя', 'хуе', 'хуи', 'хую', 'пизд', 'пизда', 'пиздец', 'пизды', 'ебать',
    'ебал', 'ебу', 'ебет', 'еби', 'еблан', 'ебаный', 'ебанат', 'блять', 'бля',
    'бляд', 'сука', 'суки', 'сучка', 'уебок', 'уебан', 'гандон', 'гондон',
    'мудак', 'мудил', 'залупа', 'ахуеть', 'охуеть', 'говно', 'гавно', 'шлюха',
    'мразь', 'долбоеб', 'выебан', 'отъебись', 'пидор', 'пидар', 'пидорас', 'чмо',
    'дрочить', 'манда', 'ебанутый', 'сперма'
]
PROFANITY_REGEX = re.compile(r'\b(' + '|'.join(PROFANITY_WORDS) + r')\w*\b', re.IGNORECASE)


# --- Вспомогательные функции ---

def pre_clean_junk(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'(?<=\w)[\.\-\*\_](?=\w)', '', text)
    text = re.sub(r'(?<=\b[a-zа-я])\s(?=[a-zа-я]\b)', '', text)
    return text


def get_champion_text_features(df_text_series):
    features = pd.DataFrame(index=df_text_series.index)
    text_as_str = df_text_series.astype(str)

    features['text_len'] = text_as_str.str.len()
    features['caps_ratio'] = text_as_str.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-6))

    words = text_as_str.str.split()
    features['word_count'] = words.str.len().fillna(0)
    features['unique_word_ratio'] = words.apply(lambda x: len(set(x)) / (len(x) + 1e-6) if isinstance(x, list) else 0)
    features['mean_word_len'] = text_as_str.apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    features['heavy_profanity_count'] = text_as_str.str.lower().str.count(PROFANITY_REGEX)
    features['non_alpha_char_count'] = text_as_str.str.count(r'[^a-zA-Zа-яА-Я0-9\s]')
    features['non_alpha_ratio'] = features['non_alpha_char_count'] / (features['text_len'] + 1e-6)

    cyrillic_chars = re.compile(r'[а-яА-Я]')
    latin_chars = re.compile(r'[a-zA-Z]')
    def count_mixed_lang_words(text):
        count = 0
        if not isinstance(text, str):
            return 0
        for word in text.split():
            if cyrillic_chars.search(word) and latin_chars.search(word):
                count += 1
        return count
    features['mixed_lang_word_count'] = text_as_str.apply(count_mixed_lang_words)
    features['profanity_density'] = features['heavy_profanity_count'] / (features['word_count'] + 1e-6)
    return features


class ToxicityEnsemble:
    def __init__(self):
        self.deob_tokenizer = None
        self.deob_model = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.lgbm_model = None
        self.meta_model = None
        self.preprocessor = None
        self.best_threshold = 0.5
        self.feature_columns = []
        self.bert_embedding_size = 768

    def _get_bert_embeddings(self, texts: list, model, tokenizer) -> np.ndarray:
        all_embeddings = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch_texts = texts[i:i + 64]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def assemble_from_checkpoints(self, df: pd.DataFrame):
        print(">>> ЗАПУСК АВАРИЙНОЙ СБОРКИ МОДЕЛИ ИЗ ЧЕКПОИНТОВ <<<")
        DEOBFUSCATED_DATA_CACHE_PATH = CACHE_DIR / 'preprocessed_data.parquet'

        if not DEOBFUSCATED_DATA_CACHE_PATH.exists():
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не могу найти кэш '{DEOBFUSCATED_DATA_CACHE_PATH}'")
            print("!!! Без него сборка невозможна. Убедитесь, что папка cache_hyperdrive на месте.")
            exit()

        print(f"\n[Шаг 1/5] Загрузка готовых данных из '{DEOBFUSCATED_DATA_CACHE_PATH}'...")
        df = pd.read_parquet(DEOBFUSCATED_DATA_CACHE_PATH)
        self.deob_tokenizer = T5Tokenizer.from_pretrained(LOCAL_DEOBFUSCATOR_PATH)
        self.deob_model = T5ForConditionalGeneration.from_pretrained(LOCAL_DEOBFUSCATOR_PATH).to(DEVICE)

        print("\n[Шаг 2/5] Сборка OOF-предсказаний из чекпоинтов...")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        oof_preds_lgbm = np.zeros(len(df))
        y_true_all = np.zeros(len(df))
        oof_bert_embeddings = np.zeros((len(df), self.bert_embedding_size))

        last_fold_bert_model_path = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            print(f"\n--- FOLD {fold + 1}/{N_SPLITS} ---")
            X_train_df, X_val_df = df.iloc[train_idx], df.iloc[val_idx]
            y_train, y_val = df['label'].iloc[train_idx].to_numpy(), df['label'].iloc[val_idx].to_numpy()
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            print(f"    scale_pos_weight: {scale_pos_weight:.2f}")

            train_eng_features = get_champion_text_features(X_train_df['cleaned_text'])
            val_eng_features = get_champion_text_features(X_val_df['cleaned_text'])
            if fold == 0:
                self.feature_columns = train_eng_features.columns.tolist()

            train_features_df = pd.concat([X_train_df, train_eng_features], axis=1)
            val_features_df = pd.concat([X_val_df, val_eng_features], axis=1)
            
            preprocessor = ColumnTransformer([
                ('word_tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000), 'cleaned_text'),
                ('char_tfidf', TfidfVectorizer(ngram_range=(3, 5), max_features=15000, analyzer='char_wb'), 'cleaned_text'),
                ('numeric', StandardScaler(), self.feature_columns)],
                remainder='drop', n_jobs=-1
            )
            
            X_train_vec = preprocessor.fit_transform(train_features_df)
            X_val_vec = preprocessor.transform(val_features_df)
            if fold == 0:
                self.preprocessor = preprocessor

            lgbm = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1000, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)
            lgbm.fit(X_train_vec, y_train, eval_set=[(X_val_vec, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
            oof_preds_lgbm[val_idx] = lgbm.predict_proba(X_val_vec)[:, 1]
            y_true_all[val_idx] = y_val

            print("    Ищем и загружаем готовый чекпоинт BERT...")
            fold_dir = CACHE_DIR / f'fold_{fold}'
            checkpoints = sorted(list(fold_dir.glob("checkpoint-*")), key=os.path.getmtime, reverse=True)
            if not checkpoints:
                raise FileNotFoundError(f"Чекпоинты для фолда {fold} не найдены в {fold_dir}!")
            
            checkpoint_path = checkpoints[0]
            print(f"    Нашел! Использую: {checkpoint_path}")

            config = PeftConfig.from_pretrained(checkpoint_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=2)
            bert_model_fold = PeftModel.from_pretrained(base_model, checkpoint_path).to(DEVICE)
            bert_tokenizer_fold = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
            last_fold_bert_model_path = checkpoint_path
            
            oof_bert_embeddings[val_idx] = self._get_bert_embeddings(X_val_df['cleaned_text'].tolist(), bert_model_fold, bert_tokenizer_fold)

        print("\n[Шаг 3/5] Обучение мета-модели (это быстро)...")
        all_eng_features = get_champion_text_features(df['cleaned_text'])
        meta_features_train = np.hstack([oof_preds_lgbm.reshape(-1, 1), oof_bert_embeddings, all_eng_features.values])
        scale_pos_weight_meta = (y_true_all == 0).sum() / (y_true_all == 1).sum()
        self.meta_model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight_meta, random_state=RANDOM_STATE).fit(meta_features_train, y_true_all)

        print("\n[Шаг 4/5] Подбор порога и финальный отчет...")
        final_oof_probs = self.meta_model.predict_proba(meta_features_train)[:, 1]
        best_f1, best_threshold = 0, 0.5
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (final_oof_probs > threshold).astype(int)
            _f1 = f1_score(y_true_all, preds, zero_division=0)
            if _f1 > best_f1:
                best_f1 = _f1
                best_threshold = threshold
        self.best_threshold = best_threshold
        
        print("\n--- Финальный Отчет по Качеству (Собранный из Частей) ---")
        final_preds = (final_oof_probs > self.best_threshold).astype(int)
        print(f"Финальный F1-score: {f1_score(y_true_all, final_preds):.4f}")
        print(f"Оптимальный порог: {self.best_threshold:.2f}")

        print("\n--- Бизнес-метрики ---")
        tn, fp, fn, tp = confusion_matrix(y_true_all, final_preds).ravel()
        print(f"Точность на чистых комментах: {tn / (tn + fp):.2%}")
        print(f"Процент пропущенного негатива: {fn / (fn + tp):.2%}")
        print(f"Процент ложных тревог: {fp / (fp + tn):.2%}")
        
        print("\n--- Полный Classification Report ---")
        print(classification_report(y_true_all, final_preds, target_names=['Нет мата (0)', 'Есть мат (1)']))

        print("\n[Шаг 5/5] Сборка финальных моделей (тоже быстро)...")
        print(" -> Обучение финального LGBM...")
        all_features_df = pd.concat([df, get_champion_text_features(df['cleaned_text'])], axis=1)
        X_full_vec = self.preprocessor.transform(all_features_df)
        scale_pos_weight_final = (df['label'] == 0).sum() / (df['label'] == 1).sum()
        self.lgbm_model = lgb.LGBMClassifier(objective='binary', n_estimators=1000, scale_pos_weight=scale_pos_weight_final, random_state=RANDOM_STATE).fit(X_full_vec, df['label'])

        print(f" -> ХАК: Используем BERT из последнего фолда ({last_fold_bert_model_path}) как финальный...")
        config = PeftConfig.from_pretrained(last_fold_bert_model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=2)
        self.bert_model = PeftModel.from_pretrained(base_model, last_fold_bert_model_path).to(DEVICE)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        print(" -> 'Запекаем' веса в основную модель...")
        self.bert_model = self.bert_model.merge_and_unload()
        print("\n>>> СБОРКА МОДЕЛИ ИЗ ЧЕКПОИНТОВ ЗАВЕРШЕНА! <<<")

    def predict(self, texts: list) -> list:
        if not all([self.bert_model, self.lgbm_model, self.meta_model, self.preprocessor]):
            raise RuntimeError("Модель не собрана!")
        
        def apply_deobfuscator_single(texts, model, tokenizer):
            cleaned_texts = []
            if not texts:
                return []
            for text in texts:
                text_lower = pre_clean_junk(str(text))
                inputs = tokenizer([text_lower], return_tensors='pt', padding=True, max_length=64, truncation=True).to(DEVICE)
                outputs = model.generate(**inputs, max_length=64, num_beams=3)
                decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                cleaned_texts.append(decoded_text)
            return cleaned_texts

        df_pred = pd.DataFrame({'text_orig': texts})
        df_pred['cleaned_text'] = apply_deobfuscator_single(df_pred['text_orig'].tolist(), self.deob_model, self.deob_tokenizer)
        eng_features_pred = get_champion_text_features(df_pred['cleaned_text'])
        features_df_pred = pd.concat([df_pred, eng_features_pred], axis=1)
        X_pred_vec = self.preprocessor.transform(features_df_pred)
        lgbm_preds = self.lgbm_model.predict_proba(X_pred_vec)[:, 1]
        bert_embeddings = self._get_bert_embeddings(df_pred['cleaned_text'].tolist(), self.bert_model, self.bert_tokenizer)
        
        meta_features_pred = np.hstack([lgbm_preds.reshape(-1, 1), bert_embeddings, eng_features_pred.values])
        final_probs = self.meta_model.predict_proba(meta_features_pred)[:, 1]
        final_labels = (final_probs > self.best_threshold).astype(int)
        
        return [{"text": txt, "label": int(lbl), "probability": float(prob)} for txt, lbl, prob in zip(texts, final_labels, final_probs)]

    def save(self, path: Path):
        print(f"\n>>> Сохранение модели в {path}...")
        path.mkdir(exist_ok=True, parents=True)
        self.deob_model.save_pretrained(path / 'deobfuscator_model')
        self.deob_tokenizer.save_pretrained(path / 'deobfuscator_model')
        self.bert_model.save_pretrained(path / 'bert_classifier')
        self.bert_tokenizer.save_pretrained(path / 'bert_classifier')
        joblib.dump(self.lgbm_model, path / 'lgbm_model.joblib')
        joblib.dump(self.meta_model, path / 'meta_model.joblib')
        joblib.dump(self.preprocessor, path / 'preprocessor.joblib')
        joblib.dump(self.best_threshold, path / 'best_threshold.joblib')
        joblib.dump(self.feature_columns, path / 'feature_columns.joblib')
        print(">>> Модель успешно сохранена.")

    @classmethod
    def load(cls, path: Path):
        print(f">>> Загрузка модели из {path}...")
        if not path.exists():
            raise FileNotFoundError(f"Папка с моделью не найдена: {path}")
            
        instance = cls()
        instance.deob_model = T5ForConditionalGeneration.from_pretrained(path / 'deobfuscator_model').to(DEVICE)
        instance.deob_tokenizer = T5Tokenizer.from_pretrained(path / 'deobfuscator_model')
        instance.bert_model = AutoModelForSequenceClassification.from_pretrained(path / 'bert_classifier').to(DEVICE)
        instance.bert_tokenizer = AutoTokenizer.from_pretrained(path / 'bert_classifier')
        instance.lgbm_model = joblib.load(path / 'lgbm_model.joblib')
        instance.meta_model = joblib.load(path / 'meta_model.joblib')
        instance.preprocessor = joblib.load(path / 'preprocessor.joblib')
        instance.best_threshold = joblib.load(path / 'best_threshold.joblib')
        instance.feature_columns = joblib.load(path / 'feature_columns.joblib')
        
        print(">>> Модель успешно загружена.")
        return instance


if __name__ == '__main__':
    if FINAL_MODEL_PATH.exists():
        print(f"Финальная модель '{FINAL_MODEL_PATH}' уже собрана. Запускаем демонстрацию.")
    else:
        print("\n" + "=" * 80)
        print("--- ЗАПУСК АВАРИЙНОЙ СБОРКИ МОДЕЛИ ИЗ ЧЕКПОИНТОВ ---")
        
        df_full = pd.read_csv('labeled2.csv')
        df_full = df_full.sample(n=min(NUM_ROWS_TO_USE, len(df_full)), random_state=RANDOM_STATE)
        df_full['label'] = df_full['label'].astype(int)
        df_full.dropna(subset=['text'], inplace=True)
        df_full = df_full.reset_index(drop=True)
        
        model = ToxicityEnsemble()
        model.assemble_from_checkpoints(df_full)
        model.save(FINAL_MODEL_PATH)
    