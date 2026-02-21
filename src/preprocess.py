# src/preprocess.py
import re
import html
from pathlib import Path
import argparse

import pandas as pd
from unidecode import unidecode
from lingua import Language, LanguageDetectorBuilder

# Optional stemming (hafif). İstersen kapalı kullan.
try:
    from nltk.stem.snowball import SnowballStemmer
    HAS_STEM = True
except Exception:
    HAS_STEM = False


ANALYZE_THRESHOLD = 0.20
MIN_ALPHA_CHARS = 8
MIN_WORDS = 2
KEEP_EMOJIS = True

TR_STOP = {
    "ve", "ile", "ama", "fakat", "çünkü", "çok", "bir", "bu", "şu", "o", "da", "de",
    "mi", "mı", "mu", "mü", "için", "gibi", "daha", "en", "şey", "ben", "sen", "biz",
    "siz", "onlar", "var", "yok", "olan", "olarak", "ya", "ki"
}
EN_STOP = {
    "the", "a", "an", "and", "or", "but", "because", "so", "to", "of", "in", "on",
    "for", "with", "is", "are", "was", "were", "be", "been", "it", "this", "that",
    "i", "you", "we", "they", "my", "your", "our", "their"
}

LANG_MAP = {
    Language.TURKISH: "tr",
    Language.ENGLISH: "en",
}


def is_low_signal(text: str) -> bool:
    if not text:
        return True
    t = str(text).strip()
    words = re.findall(r"\w+", t, flags=re.UNICODE)
    if len(words) < MIN_WORDS:
        return True
    alpha = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]", t)
    if len(alpha) < MIN_ALPHA_CHARS:
        return True
    return False


def detect_language(detector, text: str) -> str:
    if is_low_signal(text):
        return "unknown"

    langs = detector.compute_language_confidence_values(text)
    if not langs:
        return "unknown"

    top = langs[0]
    if top.value < 0.35:
        return "unknown"

    return LANG_MAP.get(top.language, "other")


def normalize_repeated_chars(s: str) -> str:
    # “çoooook” -> “çoook” gibi çok basit azaltma
    return re.sub(r"(.)\1{3,}", r"\1\1", s, flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower(), flags=re.UNICODE)


def maybe_stem(tokens: list[str], lang: str, enable: bool) -> list[str]:
    if not enable:
        return tokens
    if not HAS_STEM:
        return tokens

    if lang == "tr":
        stemmer = SnowballStemmer("turkish")
        return [stemmer.stem(t) for t in tokens]
    if lang == "en":
        stemmer = SnowballStemmer("english")
        return [stemmer.stem(t) for t in tokens]
    return tokens


def clean_text(text: str, bucket: str, enable_stem: bool) -> str:
    if text is None:
        return ""

    t = html.unescape(str(text))
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()

    t = normalize_repeated_chars(t)

    # URLs / mentions
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)

    # hashtags kelime kalsın
    t = t.replace("#", "")

    if not KEEP_EMOJIS:
        t = re.sub(r"[\U00010000-\U0010ffff]", " ", t)

    t = t.lower()
    t = re.sub(r"[^\w\sçğıöşü]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()

    tokens = tokenize(t)

    if bucket == "tr":
        tokens = [w for w in tokens if w not in TR_STOP]
        tokens = maybe_stem(tokens, "tr", enable_stem)
    elif bucket == "en":
        tokens = [unidecode(w) for w in tokens]
        tokens = [w for w in tokens if w not in EN_STOP]
        tokens = maybe_stem(tokens, "en", enable_stem)

    return " ".join(tokens)


def decide_analyze_langs(lang_counts: pd.Series, threshold: float) -> set[str]:
    total = lang_counts.sum()
    if total == 0:
        return set()

    ratios = (lang_counts / total).sort_values(ascending=False)
    analyze = set()
    for lang, r in ratios.items():
        if lang in {"unknown", "other"}:
            continue
        if r >= threshold:
            analyze.add(lang)
    return analyze


def process_file(path: Path, threshold: float, enable_stem: bool):
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(f"{path.name} içinde 'text' kolonu yok.")

    # ✅ dedup
    if "comment_id" in df.columns:
        df = df.drop_duplicates(subset=["comment_id"]).copy()

    detector = LanguageDetectorBuilder.from_languages(
        Language.TURKISH, Language.ENGLISH,
        Language.GERMAN, Language.FRENCH, Language.SPANISH,
        Language.ITALIAN, Language.PORTUGUESE,
        Language.RUSSIAN, Language.ARABIC,
        Language.JAPANESE, Language.KOREAN, Language.CHINESE,
    ).build()

    df["lang"] = df["text"].astype(str).apply(lambda x: detect_language(detector, x))

    lang_counts = df["lang"].value_counts()
    analyze_langs = decide_analyze_langs(lang_counts, threshold)

    def bucket(lang: str) -> str:
        if lang == "unknown":
            return "unknown"
        if lang in analyze_langs:
            return lang
        return "others"

    df["bucket"] = df["lang"].apply(bucket)
    df["clean_text"] = df.apply(lambda r: clean_text(r["text"], r["bucket"], enable_stem), axis=1)

    df["char_count"] = df["clean_text"].astype(str).apply(len)
    df["word_count"] = df["clean_text"].astype(str).apply(lambda x: len(x.split()) if x else 0)

    return df, analyze_langs, lang_counts


# -----------------------
# ✅ Pipeline entry-point
# -----------------------
def preprocess_run(
    run_path: Path,
    threshold: float = ANALYZE_THRESHOLD,
    stem: bool = False,
    input_name: str = "all.csv",
) -> dict[str, Path]:
    """
    Input:  <run_path>/raw/<input_name>   (default all.csv)
    Output: <run_path>/processed/{tr,en,others,unknown}.csv
    """
    raw_file = run_path / "raw" / input_name
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw input bulunamadı: {raw_file}")

    out_dir = run_path / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_stem = bool(stem) and HAS_STEM

    df, analyze_langs, lang_counts = process_file(raw_file, threshold, enable_stem)

    print("\n--- Language distribution ---")
    total = int(lang_counts.sum())
    for k, v in lang_counts.items():
        print(f"{k:8s}: {int(v):5d}  ({(v/total) if total else 0:.1%})")
    print(f"Analyze langs (>= {threshold:.0%}): {sorted(list(analyze_langs))}")
    if stem and not HAS_STEM:
        print("⚠️ NLTK yok / stemming pas geçildi. (pip install nltk)")

    outputs: dict[str, Path] = {}
    for bucket_name, sub in df.groupby("bucket"):
        out_path = out_dir / f"{bucket_name}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8")
        outputs[bucket_name] = out_path
        print(f"✅ Saved: {bucket_name:8s} -> {out_path} ({len(sub)})")

    return outputs


# -----------------------
# Optional CLI (debug)
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="İşlenecek CSV yolu (debug).")
    parser.add_argument("--threshold", type=float, default=ANALYZE_THRESHOLD, help="Dil analiz eşiği (örn 0.20)")
    parser.add_argument("--stem", action="store_true", help="TR/EN için Snowball stemming aç")
    args = parser.parse_args()

    # CLI modunda verilen file'ı işler, aynı klasöre processed yazar (debug amaçlı).
    if not args.file:
        raise RuntimeError("--file vermelisin. Pipeline için run_pipeline.py kullan.")

    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = in_path.parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    enable_stem = bool(args.stem) and HAS_STEM
    df, analyze_langs, lang_counts = process_file(in_path, args.threshold, enable_stem)

    print("\n--- Language distribution ---")
    total = int(lang_counts.sum())
    for k, v in lang_counts.items():
        print(f"{k:8s}: {int(v):5d}  ({(v/total) if total else 0:.1%})")
    print(f"Analyze langs (>= {args.threshold:.0%}): {sorted(list(analyze_langs))}")

    for bucket_name, sub in df.groupby("bucket"):
        out_path = out_dir / f"{bucket_name}.csv"
        sub.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Saved: {bucket_name:8s} -> {out_path.name} ({len(sub)})")


if __name__ == "__main__":
    main()