# src/run_pipeline.py

import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from fetch_comments import fetch_and_save, extract_video_id
from preprocess import preprocess_run
from analyze_basic import analyze_run
from analyze_sentiment import analyze_sentiment_run
from analyze_topics import analyze_topics_run
from ai_interpretation import generate_ai_topic_insights
from ai_interpretation import generate_ai_reports


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_run_folder(video_id: str) -> Path:
    root = project_root()
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}__{video_id}"

    run_path = runs_dir / run_id
    (run_path / "raw").mkdir(parents=True, exist_ok=True)
    (run_path / "processed").mkdir(parents=True, exist_ok=True)
    (run_path / "reports").mkdir(parents=True, exist_ok=True)

    return run_path


def update_meta(run_path: Path, source_url: str, params: dict):
    meta_path = run_path / "meta.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta.update(
        {
            "source_url": source_url,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "params": params,
            "artifacts": {
                "raw_all": "raw/all.csv",
                "raw_top": "raw/top.csv",
                "raw_replies": "raw/replies.csv",
                "processed": "processed/",
                "reports": "reports/",
            },
        }
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY bulunamadı.")

    url_or_id = input("YouTube linki veya video_id gir: ").strip()
    video_id = extract_video_id(url_or_id)

    params = {
        "include_replies": True,
        "order": "relevance",
        "threshold": 0.20,
        "stem": False,
        "run_basic_analysis": True,
        "run_sentiment_analysis": True,
        "run_topic_analysis": True,
        "run_ai_interpretation": True,
        "topic_clusters": 5,
    }

    print("🚀 Creating run...")
    run_path = create_run_folder(video_id)

    print("📥 Fetching comments and video metadata...")
    fetch_and_save(
        api_key=api_key,
        video_id=video_id,
        run_path=run_path,
        include_replies=params["include_replies"],
        order=params["order"],
    )

    update_meta(
        run_path=run_path,
        source_url=url_or_id,
        params=params,
    )

    print("🧹 Preprocessing...")
    preprocess_run(
        run_path=run_path,
        threshold=params["threshold"],
        stem=params["stem"],
        input_name="all.csv",
    )

    if params["run_basic_analysis"]:
        print("📊 Running basic analysis...")
        analyze_run(run_path=run_path)

    if params["run_sentiment_analysis"]:
        print("💬 Running sentiment analysis...")
        analyze_sentiment_run(run_path=run_path)

    if params["run_topic_analysis"]:
        print("🧠 Running topic analysis...")
        analyze_topics_run(
            run_path=run_path,
            n_clusters=params["topic_clusters"],
        )

    if params["run_ai_interpretation"]:
        try:
            print("🤖 Generating AI insights...")
            generate_ai_reports(run_path)
        except Exception as e:
            print(f"⚠️ AI insights skipped: {e}")

    print(f"\n✅ DONE. Results in:\n{run_path}")


if __name__ == "__main__":
    main()