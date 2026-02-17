import re
import html
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
from unidecode import unidecode
from lingua import Language, LanguageDetectorBuilder


# ---------------------------
# Config
# ---------------------------
ANALYZE_THRESHOLD = 0.20  # %20 ve üstü analiz edilecek
MIN_ALPHA_CHARS = 8       # dil tespiti için minimum harf sayısı
MIN_WORDS = 2             # çok kısa yorumları unknown yap
KEEP_EMOJIS = True        # emoji kalsın mı?

# TR/EN stopword (hafif liste; istersek büyütürüz)
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_latest_csvs(data_dir: Path) -> list[Path]:
    # data/*.csv içindeki comment dosyalarını bul
    files = sorted(data_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    # çok alakasız csv’ler varsa filtrelemek için:
    files = [f for f in files if "_top_" in f.name or "_replies_" in f.name or "_all_" in f.name]
    return files


def is_low_signal(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    # çok kısa / sadece emoji / sadece noktalama
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
    # lingua confidence 0-1 arası; çok düşükse unknown diyelim
    if top.value < 0.35:
        return "unknown"

    return LANG_MAP.get(top.language, "other")


def clean_text(text: str, lang: str) -> str:
    if text is None:
        return ""

    # HTML entity decode
    t = html.unescape(text)

    # normalize whitespace
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()

    # remove urls
    t = re.sub(r"http\S+|www\.\S+", " ", t)

    # remove mentions (YouTube'da @user)
    t = re.sub(r"@\w+", " ", t)

    # keep hashtags words but remove '#'
    t = t.replace("#", "")

    # optionally remove emojis & symbols
    if not KEEP_EMOJIS:
        t = re.sub(r"[\U00010000-\U0010ffff]", " ", t)  # emoji range (basic)

    # lowercasing (TR/EN)
    t = t.lower()

    # remove extra punct except letters/numbers/spaces
    t = re.sub(r"[^\w\sçğıöşü]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()

    # tokenize
    tokens = t.split()

    # stopwords (only for tr/en). For other languages: no stopword removal
    if lang == "tr":
        tokens = [w for w in tokens if w not in TR_STOP]
    elif lang == "en":
        # optional: normalize accents in EN
        tokens = [unidecode(w) for w in tokens]
        tokens = [w for w in tokens if w not in EN_STOP]

    return " ".join(tokens)


def decide_analyze_langs(lang_counts: pd.Series, threshold: float) -> set[str]:
    """
    lang_counts: counts by lang
    Returns languages to analyze (>= threshold), excluding unknown/other.
    """
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


def process_file(path: Path, threshold: float):
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(f"{path.name} içinde 'text' kolonu yok.")

    # lingua detector (TR+EN + "rest" için auto detect; burada TR/EN hedef, diğerleri other)
    detector = LanguageDetectorBuilder.from_languages(
        Language.TURKISH, Language.ENGLISH,
        Language.GERMAN, Language.FRENCH, Language.SPANISH,
        Language.ITALIAN, Language.PORTUGUESE,
        Language.RUSSIAN, Language.ARABIC,
        Language.JAPANESE, Language.KOREAN, Language.CHINESE,
    ).build()

    df["lang"] = df["text"].astype(str).apply(lambda x: detect_language(detector, x))

    # dil dağılımı
    lang_counts = df["lang"].value_counts()
    analyze_langs = decide_analyze_langs(lang_counts, threshold)

    # bucket
    def bucket(lang: str) -> str:
        if lang == "unknown":
            return "unknown"
        if lang in analyze_langs:
            return lang
        return "others"

    df["bucket"] = df["lang"].apply(bucket)

    # clean_text
    df["clean_text"] = df.apply(lambda r: clean_text(str(r["text"]), r["bucket"]), axis=1)

    # basic stats
    df["char_count"] = df["clean_text"].astype(str).apply(len)
    df["word_count"] = df["clean_text"].astype(str).apply(lambda x: len(x.split()) if x else 0)

    return df, analyze_langs, lang_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="", help="İşlenecek CSV yolu. Boşsa data/ içinden en güncelini alır.")
    parser.add_argument("--threshold", type=float, default=ANALYZE_THRESHOLD, help="Analiz edilecek dil eşiği (örn 0.20)")
    args = parser.parse_args()

    root = project_root()
    data_dir = root / "data"
    out_dir = data_dir / "processed"
    out_dir.mkdir(exist_ok=True)

    if args.file:
        files = [Path(args.file)]
    else:
        files = load_latest_csvs(data_dir)
        if not files:
            raise RuntimeError("data/ altında *_top_ / *_replies_ / *_all_ CSV bulunamadı.")

        # sadece en güncel 3’lü seti işlemek istersen burada kırpabiliriz
        # şimdilik en güncel dosyayı al:
        files = [files[0]]

    for f in files:
        df, analyze_langs, lang_counts = process_file(f, args.threshold)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f.stem  # örn video_all_2026...

        # rapor
        print("\n--- Language distribution ---")
        total = lang_counts.sum()
        for k, v in lang_counts.items():
            print(f"{k:8s}: {v:5d}  ({v/total:.1%})")
        print(f"Analyze langs (>= {args.threshold:.0%}): {sorted(list(analyze_langs))}")

        # çıktıları bucket bazlı kaydet
        for bucket_name, sub in df.groupby("bucket"):
            out_path = out_dir / f"{base}__{bucket_name}__{stamp}.csv"
            sub.to_csv(out_path, index=False, encoding="utf-8")
            print(f"✅ Saved: {bucket_name:8s} -> {out_path.name} ({len(sub)})")


if __name__ == "__main__":
    main()