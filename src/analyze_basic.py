# src/analyze_basic.py
from pathlib import Path
import re
import pandas as pd


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower(), flags=re.UNICODE)


def ngrams(tokens: list[str], n=2) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def analyze_one(df: pd.DataFrame, reports_dir: Path, tag: str) -> dict[str, Path]:
    """
    Writes files with stable names inside a run's reports folder.
    """
    if "clean_text" not in df.columns:
        raise ValueError(f"{tag}: 'clean_text' yok. preprocess doğru mu?")

    reports_dir.mkdir(parents=True, exist_ok=True)

    n_comments = len(df)
    n_authors = df["author"].nunique() if "author" in df.columns else None
    avg_words = (
        df["word_count"].mean()
        if "word_count" in df.columns
        else df["clean_text"].astype(str).apply(lambda x: len(x.split())).mean()
    )
    avg_chars = (
        df["char_count"].mean()
        if "char_count" in df.columns
        else df["clean_text"].astype(str).apply(len).mean()
    )

    # vocab size / TTR
    all_tokens: list[str] = []
    for t in df["clean_text"].astype(str):
        all_tokens.extend(tokenize(t))
    vocab = set(all_tokens)
    ttr = (len(vocab) / len(all_tokens)) if all_tokens else 0.0

    word_freq = pd.Series(all_tokens).value_counts().head(50).reset_index()
    word_freq.columns = ["word", "count"]

    all_bigrams: list[str] = []
    for t in df["clean_text"].astype(str):
        toks = tokenize(t)
        all_bigrams.extend(ngrams(toks, 2))
    bigram_freq = pd.Series(all_bigrams).value_counts().head(50).reset_index()
    bigram_freq.columns = ["bigram", "count"]

    # Most liked
    top_like = pd.DataFrame()
    if "like_count" in df.columns:
        top_like = df.sort_values("like_count", ascending=False).head(15)[
            [c for c in ["like_count", "published_at", "author", "text", "clean_text"] if c in df.columns]
        ]

    # Time distribution (daily)
    time_summary = pd.DataFrame()
    if "published_at" in df.columns:
        tmp = df.copy()
        tmp["published_at"] = pd.to_datetime(tmp["published_at"], errors="coerce", utc=True)
        tmp = tmp.dropna(subset=["published_at"])
        tmp["date"] = tmp["published_at"].dt.date
        time_summary = tmp.groupby("date").size().reset_index(name="comment_count").sort_values("date")

    # ✅ Save with stable names
    word_path = reports_dir / f"{tag}_top_words.csv"
    bigram_path = reports_dir / f"{tag}_top_bigrams.csv"
    report_path = reports_dir / f"{tag}_basic_report.txt"

    word_freq.to_csv(word_path, index=False, encoding="utf-8")
    bigram_freq.to_csv(bigram_path, index=False, encoding="utf-8")

    top_like_path = None
    if not top_like.empty:
        top_like_path = reports_dir / f"{tag}_top_liked.csv"
        top_like.to_csv(top_like_path, index=False, encoding="utf-8")

    time_path = None
    if not time_summary.empty:
        time_path = reports_dir / f"{tag}_time_distribution.csv"
        time_summary.to_csv(time_path, index=False, encoding="utf-8")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {tag.upper()} REPORT ===\n")
        f.write(f"Comments: {n_comments}\n")
        if n_authors is not None:
            f.write(f"Unique authors: {n_authors}\n")
        f.write(f"Avg words: {avg_words:.2f}\n")
        f.write(f"Avg chars: {avg_chars:.2f}\n")
        f.write(f"Vocab size: {len(vocab)}\n")
        f.write(f"Type-Token Ratio: {ttr:.4f}\n\n")

        f.write("OUTPUT FILES:\n")
        f.write(f"- {word_path.name}\n")
        f.write(f"- {bigram_path.name}\n")
        if top_like_path:
            f.write(f"- {top_like_path.name}\n")
        if time_path:
            f.write(f"- {time_path.name}\n")

    return {
        "report": report_path,
        "top_words": word_path,
        "top_bigrams": bigram_path,
        "top_liked": top_like_path,
        "time_dist": time_path,
    }


# -----------------------
# ✅ Pipeline entry-point
# -----------------------
def analyze_run(run_path: Path) -> dict[str, dict[str, Path]]:
    """
    Input:  <run_path>/processed/{tr,en,others,unknown}.csv  (varsa)
    Output: <run_path>/reports/*
    """
    processed_dir = run_path / "processed"
    reports_dir = run_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    buckets = ["tr", "en", "others", "unknown"]
    results: dict[str, dict[str, Path]] = {}

    found_any = False
    for b in buckets:
        in_path = processed_dir / f"{b}.csv"
        if not in_path.exists():
            continue
        found_any = True

        df = pd.read_csv(in_path)
        out = analyze_one(df, reports_dir, tag=b)
        results[b] = out
        print(f"✅ {b} done -> {out['report'].name}")

    if not found_any:
        raise RuntimeError(f"Processed dosyası bulunamadı: {processed_dir} içinde tr/en/others/unknown.csv yok.")

    return results


# Optional CLI (debug)
def main():
    raise RuntimeError("Bu dosyayı tek başına çalıştırma. Pipeline için: python src/run_pipeline.py")


if __name__ == "__main__":
    main()