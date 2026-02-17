from pathlib import Path
from datetime import datetime
import re
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def latest_processed_tr(processed_dir: Path) -> Path:
    files = sorted(processed_dir.glob("*__tr__*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("data/processed içinde '__tr__' dosyası bulunamadı.")
    return files[0]


def tokenize(text: str):
    if not isinstance(text, str):
        return []
    # kelime token
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower())


def ngrams(tokens, n=2):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def main():
    root = project_root()
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    tr_file = latest_processed_tr(processed_dir)
    df = pd.read_csv(tr_file)

    # Kolonlar
    if "clean_text" not in df.columns:
        raise ValueError("Bu dosyada 'clean_text' yok. preprocess doğru çalışmış mı?")

    # Basic counts
    n_comments = len(df)
    n_authors = df["author"].nunique() if "author" in df.columns else None
    avg_words = df["word_count"].mean() if "word_count" in df.columns else df["clean_text"].astype(str).apply(lambda x: len(x.split())).mean()
    avg_chars = df["char_count"].mean() if "char_count" in df.columns else df["clean_text"].astype(str).apply(len).mean()

    # Top words
    all_tokens = []
    for t in df["clean_text"].astype(str):
        all_tokens.extend(tokenize(t))

    word_freq = pd.Series(all_tokens).value_counts().head(50).reset_index()
    word_freq.columns = ["word", "count"]

    # Top bigrams
    all_bigrams = []
    for t in df["clean_text"].astype(str):
        toks = tokenize(t)
        all_bigrams.extend(ngrams(toks, 2))
    bigram_freq = pd.Series(all_bigrams).value_counts().head(50).reset_index()
    bigram_freq.columns = ["bigram", "count"]

    # Most liked comments
    if "like_count" in df.columns:
        top_like = df.sort_values("like_count", ascending=False).head(15)[
            [c for c in ["like_count", "published_at", "author", "text", "clean_text"] if c in df.columns]
        ]
    else:
        top_like = pd.DataFrame()

    # Time distribution (daily)
    time_summary = pd.DataFrame()
    if "published_at" in df.columns:
        tmp = df.copy()
        tmp["published_at"] = pd.to_datetime(tmp["published_at"], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=["published_at"])
        tmp["date"] = tmp["published_at"].dt.date
        time_summary = tmp.groupby("date").size().reset_index(name="comment_count").sort_values("date")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save outputs
    word_path = reports_dir / f"top_words_{stamp}.csv"
    bigram_path = reports_dir / f"top_bigrams_{stamp}.csv"
    word_freq.to_csv(word_path, index=False, encoding="utf-8")
    bigram_freq.to_csv(bigram_path, index=False, encoding="utf-8")

    if not top_like.empty:
        top_like_path = reports_dir / f"top_liked_comments_{stamp}.csv"
        top_like.to_csv(top_like_path, index=False, encoding="utf-8")
    else:
        top_like_path = None

    if not time_summary.empty:
        time_path = reports_dir / f"time_distribution_{stamp}.csv"
        time_summary.to_csv(time_path, index=False, encoding="utf-8")
    else:
        time_path = None

    report_path = reports_dir / f"basic_report_{stamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"INPUT FILE: {tr_file.name}\n\n")
        f.write("=== BASIC STATS ===\n")
        f.write(f"Comments: {n_comments}\n")
        if n_authors is not None:
            f.write(f"Unique authors: {n_authors}\n")
        f.write(f"Avg words: {avg_words:.2f}\n")
        f.write(f"Avg chars: {avg_chars:.2f}\n\n")

        f.write("=== OUTPUT FILES ===\n")
        f.write(f"Top words: {word_path.name}\n")
        f.write(f"Top bigrams: {bigram_path.name}\n")
        if top_like_path:
            f.write(f"Top liked comments: {top_like_path.name}\n")
        if time_path:
            f.write(f"Time distribution: {time_path.name}\n")

    # Print summary
    print("\n✅ BASIC EDA DONE")
    print(f"Input: {tr_file}")
    print(f"Report: {report_path}")
    print(f"Top words: {word_path}")
    print(f"Top bigrams: {bigram_path}")
    if top_like_path:
        print(f"Top liked: {top_like_path}")
    if time_path:
        print(f"Time dist: {time_path}")


if __name__ == "__main__":
    main()