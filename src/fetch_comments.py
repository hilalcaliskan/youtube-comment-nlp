# src/fetch_comments.py

import os
import re
import json
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    if "youtu.be/" in s:
        return s.split("youtu.be/")[-1].split("?")[0].split("&")[0]

    parsed = urlparse(s)
    qs = parse_qs(parsed.query)

    if "v" in qs and qs["v"]:
        return qs["v"][0]

    if "/shorts/" in s:
        return s.split("/shorts/")[-1].split("?")[0].split("&")[0]

    raise ValueError("Video ID bulunamadı.")


def fetch_video_metadata(youtube, video_id: str) -> dict:
    req = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )

    res = req.execute()

    items = res.get("items", [])

    if not items:
        raise RuntimeError("Video metadata alınamadı.")

    item = items[0]

    snippet = item["snippet"]
    stats = item["statistics"]

    thumbnails = snippet.get("thumbnails", {})

    thumbnail_url = (
        thumbnails.get("maxres", {}).get("url")
        or thumbnails.get("high", {}).get("url")
        or thumbnails.get("medium", {}).get("url")
        or thumbnails.get("default", {}).get("url")
    )

    return {
        "video_id": video_id,
        "title": snippet.get("title"),
        "channel_title": snippet.get("channelTitle"),
        "published_at": snippet.get("publishedAt"),
        "thumbnail_url": thumbnail_url,
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
    }


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
    order: str = "relevance",
):
    youtube = build("youtube", "v3", developerKey=api_key)

    metadata = fetch_video_metadata(youtube, video_id)

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

            rows.append(
                {
                    "comment_id": top_id,
                    "parent_id": None,
                    "like_count": top.get("likeCount", 0),
                    "published_at": top.get("publishedAt"),
                    "text": top.get("textDisplay", ""),
                }
            )

            reply_count = item["snippet"].get("totalReplyCount", 0)

            if include_replies and reply_count > 0:
                try:
                    rows.extend(fetch_replies(youtube, top_id))

                except HttpError as e:
                    print(f"⚠️ Replies alınamadı: {e}")

        next_page_token = res.get("nextPageToken")

        if not next_page_token:
            break

    df = pd.DataFrame(rows)

    return df, metadata


def save_three_csvs(df: pd.DataFrame, run_path: Path):

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

    print(f"✅ Top-level: {len(top_df)}")
    print(f"✅ Replies: {len(rep_df)}")
    print(f"✅ Total: {len(df)}")

    return {
        "top": top_path,
        "replies": rep_path,
        "all": all_path,
    }


def save_metadata(metadata: dict, run_path: Path):

    meta_path = run_path / "meta.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("✅ Video metadata saved")


def fetch_and_save(
    api_key: str,
    video_id: str,
    run_path: Path,
    include_replies: bool = True,
    order: str = "relevance",
):

    df, metadata = fetch_all_comments(
        api_key=api_key,
        video_id=video_id,
        include_replies=include_replies,
        order=order,
    )

    save_metadata(metadata, run_path)

    return save_three_csvs(df, run_path)


def main():

    load_dotenv()

    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY bulunamadı.")

    url_or_id = input("YouTube linki veya video_id gir: ").strip()

    video_id = extract_video_id(url_or_id)

    project_root = Path(__file__).resolve().parents[1]

    runs_dir = project_root / "runs"
    runs_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_path = runs_dir / f"{stamp}__{video_id}"

    (run_path / "raw").mkdir(parents=True, exist_ok=True)

    fetch_and_save(
        api_key,
        video_id,
        run_path,
        include_replies=True,
        order="relevance",
    )


if __name__ == "__main__":
    main()