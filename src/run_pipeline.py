# src/run_pipeline.py
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from fetch_comments import fetch_and_save, extract_video_id
from preprocess import preprocess_run
from analyze_basic import analyze_run


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


def save_meta(run_path: Path, video_id: str, source_url: str, params: dict):
    meta = {
        "video_id": video_id,
        "source_url": source_url,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "params": params,
        "artifacts": {
            "raw_all": "raw/all.csv",
            "raw_top": "raw/top.csv",
            "raw_replies": "raw/replies.csv",
            # processed ve reports sonradan oluşacak (istersen en sonunda güncelleriz)
        },
    }
    with open(run_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY bulunamadı.")

    url_or_id = input("YouTube linki veya video_id gir: ").strip()
    video_id = extract_video_id(url_or_id)

    # ✅ Pipeline params (ileride cache için de kullanılacak)
    params = {
        "include_replies": True,
        "order": "relevance",   # istersen "time"
        "threshold": 0.20,
        "stem": False,
    }

    print("🚀 Creating run...")
    run_path = create_run_folder(video_id)
    save_meta(run_path, video_id, source_url=url_or_id, params=params)

    print("📥 Fetching comments...")
    fetch_and_save(
        api_key=api_key,
        video_id=video_id,
        run_path=run_path,
        include_replies=params["include_replies"],
        order=params["order"],
    )

    print("🧹 Preprocessing...")
    preprocess_run(
        run_path=run_path,
        threshold=params["threshold"],
        stem=params["stem"],
        input_name="all.csv",
    )

    print("📊 Analyzing...")
    analyze_run(run_path=run_path)

    print(f"\n✅ DONE. Results in:\n{run_path}")


if __name__ == "__main__":
    main()