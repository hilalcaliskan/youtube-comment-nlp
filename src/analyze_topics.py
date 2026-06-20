# src/analyze_topics.py

from pathlib import Path
from typing import Dict, Set

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TR_STOPWORDS = {
    "ve", "bir", "bu", "şu", "o", "ile", "ama", "fakat", "çünkü",
    "çok", "cok", "daha", "en", "gibi", "için", "olan", "olarak",
    "ya", "yani", "bence", "zaten", "artık", "şey", "kadar",
    "ne", "ki", "da", "de", "mi", "mı", "mu", "mü",
    "var", "yok", "hem", "her", "hiç", "hep",
    "burada", "orada", "şurada",
    "ben", "sen", "biz", "siz", "onlar",
    "bana", "sana", "bize", "size",
    "benim", "senin", "bizim", "sizin",
    "abi", "abla", "adam", "kadın",
    "video", "yorum", "geldim", "izledim",
    "bak", "işte", "falan", "filan",
    "iyi", "güzel", "kötü", "değil", "bunu", "neden", "bile", "sadece", "öyle",
    "diye", "olmuş", "olsun", "oldu", "biri", "zaman",
    "son", "böyle", "gerçekten", "herkes", "insan",
}

EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "because", "so",
    "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "these", "those",
    "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their",
    "me", "him", "her", "them",
    "as", "at", "by", "from",
    "about", "into", "over", "after",
    "again", "once", "here", "there",
    "why", "how", "all", "any", "both",
    "each", "few", "more", "most",
    "other", "some", "such",
    "no", "nor", "not",
    "only", "own", "same",
    "too", "very",
}

STOPWORDS_BY_BUCKET: Dict[str, Set[str]] = {
    "tr": TR_STOPWORDS,
    "en": EN_STOPWORDS,
    "others": set(),
}


def clean_for_topic(text: str, stopwords: Set[str]) -> str:
    if not isinstance(text, str):
        return ""

    tokens = text.lower().split()
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stopwords]

    return " ".join(tokens)

def get_dynamic_stopwords(texts, top_n=15):
    """
    Finds globally overused words in this specific video.
    These words are removed from topic keywords.
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
    )

    X = vectorizer.fit_transform(texts)

    scores = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    top_indices = scores.argsort()[::-1][:top_n]

    return {words[i] for i in top_indices}

def get_cluster_keywords(df: pd.DataFrame, cluster_id: int, n_words: int = 10) -> str:
    cluster_df = df[df["topic_cluster"] == cluster_id]

    texts = cluster_df["topic_text"].fillna("").astype(str).tolist()

    if len(texts) < 2:
        return ""

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.90,
    )

    X = vectorizer.fit_transform(texts)
    scores = X.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()
    top_indices = scores.argsort()[::-1][:n_words]

    return ", ".join(words[i] for i in top_indices)


def run_topic_for_bucket(
    run_path: Path,
    bucket: str,
    n_clusters: int = 5,
    min_words: int = 4,
):
    input_path = run_path / "processed" / f"{bucket}.csv"

    if not input_path.exists():
        print(f"⚠️ {bucket}.csv bulunamadı, topic analysis atlandı.")
        return None

    df = pd.read_csv(input_path)

    if "clean_text" not in df.columns:
        print(f"⚠️ {bucket}.csv içinde clean_text yok, topic analysis atlandı.")
        return None

    if "word_count" in df.columns:
        df = df[df["word_count"] >= min_words].copy()

    if df.empty:
        print(f"⚠️ {bucket}: Topic analysis için yeterli yorum yok.")
        return None

    stopwords = STOPWORDS_BY_BUCKET.get(bucket, set())

    dynamic_stopwords = get_dynamic_stopwords(
        df["clean_text"].fillna("").astype(str).tolist(),
        top_n=12
    )

    stopwords = stopwords.union(dynamic_stopwords)

    df["topic_text"] = (
        df["clean_text"]
        .fillna("")
        .astype(str)
        .apply(lambda x: clean_for_topic(x, stopwords))
    )

    df = df[df["topic_text"].str.split().apply(len) >= 3].copy()

    if len(df) < 5:
        print(f"⚠️ {bucket}: Çok az yorum kaldı, topic analysis atlandı.")
        return None

    actual_clusters = min(n_clusters, max(2, len(df) // 25))

    if len(df) < actual_clusters:
        print(f"⚠️ {bucket}: Cluster için yeterli yorum yok.")
        return None

    text_col = "text" if "text" in df.columns else "clean_text"
    embedding_texts = df[text_col].fillna("").astype(str).tolist()

    print(f"🧠 Creating embeddings for {bucket} ({len(df)} comments)...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clustering_model = KMeans(
        n_clusters=actual_clusters,
        random_state=42,
        n_init=10,
    )

    df["topic_cluster"] = clustering_model.fit_predict(embeddings)

    reports_dir = run_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cluster_id in sorted(df["topic_cluster"].unique()):
        cluster_df = df[df["topic_cluster"] == cluster_id].copy()

        if "like_count" in cluster_df.columns:
            cluster_df = cluster_df.sort_values("like_count", ascending=False)

        examples = cluster_df[text_col].head(3).tolist()

        keywords = get_cluster_keywords(df, cluster_id, n_words=10)

        summary_rows.append(
            {
                "bucket": bucket,
                "cluster_id": cluster_id,
                "comment_count": len(cluster_df),
                "top_keywords": keywords,
                "example_1": examples[0] if len(examples) > 0 else "",
                "example_2": examples[1] if len(examples) > 1 else "",
                "example_3": examples[2] if len(examples) > 2 else "",
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = reports_dir / f"{bucket}_topic_summary.csv"
    detail_path = reports_dir / f"{bucket}_topic_details.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    df.to_csv(detail_path, index=False, encoding="utf-8")

    print(f"✅ semantic topic done for {bucket} -> {summary_path.name}")
    print(f"✅ topic details saved for {bucket} -> {detail_path.name}")

    return {
        "summary": summary_path,
        "details": detail_path,
    }


def analyze_topics_run(run_path: Path, n_clusters: int = 5):
    buckets = ["tr", "en", "others"]
    results = {}

    for bucket in buckets:
        result = run_topic_for_bucket(
            run_path=run_path,
            bucket=bucket,
            n_clusters=n_clusters,
        )

        if result is not None:
            results[bucket] = result

    if not results:
        print("⚠️ Hiçbir bucket için topic analysis üretilemedi.")

    return results