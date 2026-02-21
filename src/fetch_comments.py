# src/fetch_comments.py
import os
import re
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()

    # Raw video_id (usually 11 chars)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    # youtu.be/<id>
    if "youtu.be/" in s:
        return s.split("youtu.be/")[-1].split("?")[0].split("&")[0]

    # youtube.com/watch?v=<id>
    parsed = urlparse(s)
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]

    # youtube.com/shorts/<id>
    if "/shorts/" in s:
        return s.split("/shorts/")[-1].split("?")[0].split("&")[0]

    raise ValueError("Video ID bulunamadı. Linki veya 11 haneli video_id'yi kontrol et.")


def fetch_replies(youtube, parent_id: str) -> list[dict]:
    rows = []
    next_page_token = None

    while True:
        req = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText",
        )
        res = req.execute()

        for rep in res.get("items", []):
            snip = rep["snippet"]
            rows.append(
                {
                    "comment_id": rep["id"],
                    "parent_id": parent_id,
                    "author": snip.get("authorDisplayName"),
                    "like_count": snip.get("likeCount", 0),
                    "published_at": snip.get("publishedAt"),
                    "text": snip.get("textDisplay", ""),
                }
            )

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    return rows


def fetch_all_comments(
    api_key: str,
    video_id: str,
    include_replies: bool = True,
    order: str = "relevance",  # "time" (newest) or "relevance" (top)
) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)

    rows = []
    next_page_token = None

    while True:
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order=order,
            )
            res = req.execute()
        except HttpError as e:
            raise RuntimeError(f"YouTube API error: {e}") from e

        for item in res.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            top_id = item["snippet"]["topLevelComment"]["id"]

            # Top-level comment
            rows.append(
                {
                    "comment_id": top_id,
                    "parent_id": None,
                    "author": top.get("authorDisplayName"),
                    "like_count": top.get("likeCount", 0),
                    "published_at": top.get("publishedAt"),
                    "text": top.get("textDisplay", ""),
                }
            )

            # Replies
            reply_count = item["snippet"].get("totalReplyCount", 0)
            if include_replies and reply_count > 0:
                try:
                    rows.extend(fetch_replies(youtube, top_id))
                except HttpError as e:
                    print(f"⚠️ Replies alınamadı (parent_id={top_id}): {e}")

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(rows)


def save_three_csvs(df: pd.DataFrame, run_path: Path) -> dict[str, Path]:
    """
    Save into:
      <run_path>/raw/top.csv
      <run_path>/raw/replies.csv
      <run_path>/raw/all.csv
    Returns the written paths.
    """
    raw_dir = run_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    top_df = df[df["parent_id"].isna()].copy()
    rep_df = df[df["parent_id"].notna()].copy()

    top_path = raw_dir / "top.csv"
    rep_path = raw_dir / "replies.csv"
    all_path = raw_dir / "all.csv"

    top_df.to_csv(top_path, index=False, encoding="utf-8")
    rep_df.to_csv(rep_path, index=False, encoding="utf-8")
    df.to_csv(all_path, index=False, encoding="utf-8")

    print(f"✅ Top-level: {len(top_df)} -> {top_path}")
    print(f"✅ Replies:   {len(rep_df)} -> {rep_path}")
    print(f"✅ Total:     {len(df)} -> {all_path}")

    return {"top": top_path, "replies": rep_path, "all": all_path}


def fetch_and_save(
    api_key: str,
    video_id: str,
    run_path: Path,
    include_replies: bool = True,
    order: str = "relevance",
) -> dict[str, Path]:
    """
    Pipeline entry-point.
    """
    df = fetch_all_comments(
        api_key=api_key,
        video_id=video_id,
        include_replies=include_replies,
        order=order,
    )
    return save_three_csvs(df, run_path)


def main():
    """
    Opsiyonel CLI kullanım (run_pipeline olmadan).
    Bu modda da runs klasörü altında tek run üretir.
    """
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY bulunamadı. .env dosyasına eklediğinden emin ol.")

    url_or_id = input("YouTube linki veya video_id gir: ").strip()
    video_id = extract_video_id(url_or_id)

    # Basit run folder
    project_root = Path(__file__).resolve().parents[1]
    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = runs_dir / f"{stamp}__{video_id}"
    (run_path / "raw").mkdir(parents=True, exist_ok=True)

    fetch_and_save(api_key, video_id, run_path, include_replies=True, order="relevance")


if __name__ == "__main__":
    main()