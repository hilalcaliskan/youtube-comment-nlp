# src/analyze_sentiment.py

from pathlib import Path
from typing import Dict, List
import pandas as pd
from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
_classifier_singleton = None


def load_sentiment_pipeline():
    """
    Loads Hugging Face sentiment pipeline once.
    """
    global _classifier_singleton

    if _classifier_singleton is None:
        _classifier_singleton = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
        )

    return _classifier_singleton


def map_label(raw_label: str) -> str:
    """
    Normalize model labels to positive / neutral / negative.
    """
    label = str(raw_label).lower().strip()

    if "positive" in label:
        return "positive"
    if "negative" in label:
        return "negative"
    if "neutral" in label:
        return "neutral"

    if label in {"5 stars", "4 stars"}:
        return "positive"
    if label == "3 stars":
        return "neutral"
    if label in {"1 star", "2 stars"}:
        return "negative"

    return label


def chunk_list(items: List[str], batch_size: int) -> List[List[str]]:
    """
    Splits a list into smaller batches.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def predict_sentiment_dataframe(
    df: pd.DataFrame,
    classifier,
    text_col: str = "text",
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Adds sentiment_label and sentiment_score columns to a dataframe.
    """
    if text_col not in df.columns:
        raise ValueError(f"'{text_col}' kolonu bulunamadı.")

    out = df.copy()
    texts = out[text_col].fillna("").astype(str).tolist()

    labels = []
    scores = []

    for batch in chunk_list(texts, batch_size):
        results = classifier(batch, truncation=True, max_length=512)

        for result in results:
            labels.append(map_label(result["label"]))
            scores.append(float(result["score"]))

    out["sentiment_label"] = labels
    out["sentiment_score"] = scores

    return out


def predict_single_sentiment(text: str):
    """
    Predict sentiment for a single text.
    Used mainly for evaluation scripts.

    Returns:
        (label, score)
    """
    classifier = load_sentiment_pipeline()
    result = classifier([str(text)], truncation=True, max_length=512)[0]

    label = map_label(result["label"])
    score = float(result["score"])

    return label, score


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates sentiment distribution summary.
    """
    summary = (
        df["sentiment_label"]
        .value_counts(dropna=False)
        .rename_axis("sentiment_label")
        .reset_index(name="count")
    )

    total = len(df)
    summary["ratio"] = summary["count"] / total if total else 0

    return summary


def top_comments_by_sentiment(
    df: pd.DataFrame,
    sentiment: str,
    n: int = 10
) -> pd.DataFrame:
    """
    Returns top liked comments for a given sentiment class.
    """
    sub = df[df["sentiment_label"] == sentiment].copy()

    if sub.empty:
        return pd.DataFrame()

    sort_cols = []
    if "like_count" in sub.columns:
        sort_cols.append("like_count")

    sort_cols.append("sentiment_score")

    sub = sub.sort_values(sort_cols, ascending=False)

    keep_cols = [
        c for c in [
            "author",
            "text",
            "clean_text",
            "like_count",
            "published_at",
            "sentiment_label",
            "sentiment_score",
        ]
        if c in sub.columns
    ]

    return sub[keep_cols].head(n)


def sentiment_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily sentiment counts over time.
    """
    if "published_at" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["published_at"] = pd.to_datetime(
        tmp["published_at"],
        errors="coerce",
        utc=True
    )

    tmp = tmp.dropna(subset=["published_at"])

    if tmp.empty:
        return pd.DataFrame()

    tmp["date"] = tmp["published_at"].dt.date

    grouped = (
        tmp.groupby(["date", "sentiment_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["date", "sentiment_label"])
    )

    return grouped


def analyze_sentiment_file(
    input_path: Path,
    reports_dir: Path,
    processed_dir: Path,
    tag: str,
    classifier
) -> Dict[str, Path]:
    """
    Reads one processed CSV, predicts sentiment, saves enriched data and reports.
    """
    df = pd.read_csv(input_path)

    # Transformer modellerde ham text genelde daha iyi sonuç verir.
    text_col = "text" if "text" in df.columns else "clean_text"

    enriched = predict_sentiment_dataframe(
        df=df,
        classifier=classifier,
        text_col=text_col,
        batch_size=16,
    )

    processed_out = processed_dir / f"{tag}_sentiment.csv"
    enriched.to_csv(processed_out, index=False, encoding="utf-8")

    summary = build_summary(enriched)
    summary_path = reports_dir / f"{tag}_sentiment_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    top_positive = top_comments_by_sentiment(
        enriched,
        sentiment="positive",
        n=10
    )
    top_negative = top_comments_by_sentiment(
        enriched,
        sentiment="negative",
        n=10
    )
    top_neutral = top_comments_by_sentiment(
        enriched,
        sentiment="neutral",
        n=10
    )

    top_positive_path = None
    top_negative_path = None
    top_neutral_path = None

    if not top_positive.empty:
        top_positive_path = reports_dir / f"{tag}_top_positive_comments.csv"
        top_positive.to_csv(top_positive_path, index=False, encoding="utf-8")

    if not top_negative.empty:
        top_negative_path = reports_dir / f"{tag}_top_negative_comments.csv"
        top_negative.to_csv(top_negative_path, index=False, encoding="utf-8")

    if not top_neutral.empty:
        top_neutral_path = reports_dir / f"{tag}_top_neutral_comments.csv"
        top_neutral.to_csv(top_neutral_path, index=False, encoding="utf-8")

    time_dist = sentiment_time_distribution(enriched)
    time_dist_path = None

    if not time_dist.empty:
        time_dist_path = reports_dir / f"{tag}_sentiment_time_distribution.csv"
        time_dist.to_csv(time_dist_path, index=False, encoding="utf-8")

    report_path = reports_dir / f"{tag}_sentiment_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {tag.upper()} SENTIMENT REPORT ===\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total comments: {len(enriched)}\n\n")

        f.write("Sentiment distribution:\n")
        for _, row in summary.iterrows():
            f.write(
                f"- {row['sentiment_label']}: "
                f"{row['count']} ({row['ratio']:.1%})\n"
            )

        f.write("\nOUTPUT FILES:\n")
        f.write(f"- {processed_out.name}\n")
        f.write(f"- {summary_path.name}\n")

        if top_positive_path:
            f.write(f"- {top_positive_path.name}\n")
        if top_negative_path:
            f.write(f"- {top_negative_path.name}\n")
        if top_neutral_path:
            f.write(f"- {top_neutral_path.name}\n")
        if time_dist_path:
            f.write(f"- {time_dist_path.name}\n")

    return {
        "processed_sentiment": processed_out,
        "summary": summary_path,
        "top_positive": top_positive_path,
        "top_negative": top_negative_path,
        "top_neutral": top_neutral_path,
        "time_dist": time_dist_path,
        "report": report_path,
    }


def analyze_sentiment_run(run_path: Path) -> Dict[str, Dict[str, Path]]:
    """
    Runs sentiment analysis for processed bucket files.

    Input:
        <run_path>/processed/{tr,en,others}.csv

    Output:
        enriched sentiment CSV files under processed/
        sentiment reports under reports/
    """
    processed_dir = run_path / "processed"
    reports_dir = run_path / "reports"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    classifier = load_sentiment_pipeline()

    buckets = ["tr", "en", "others"]
    results: Dict[str, Dict[str, Path]] = {}

    found_any = False

    for tag in buckets:
        input_path = processed_dir / f"{tag}.csv"

        if not input_path.exists():
            continue

        found_any = True

        output = analyze_sentiment_file(
            input_path=input_path,
            reports_dir=reports_dir,
            processed_dir=processed_dir,
            tag=tag,
            classifier=classifier,
        )

        results[tag] = output
        print(f"✅ sentiment done for {tag} -> {output['report'].name}")

    if not found_any:
        raise RuntimeError(
            f"Sentiment için uygun processed dosyası bulunamadı: {processed_dir}"
        )

    return results


def main():
    raise RuntimeError(
        "Bu dosyayı tek başına çalıştırma. Pipeline için: python src/run_pipeline.py"
    )


if __name__ == "__main__":
    main()