# src/ai_interpretation.py

import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4.1-mini"


def get_output_language(bucket: str) -> str:
    language_map = {
        "tr": "Turkish",
        "en": "English",
        "others": "English",
    }

    return language_map.get(bucket, "English")


def clean_json_response(content: str) -> str:
    """
    Cleans responses like:
    ```json
    {...}
    ```
    and extracts the JSON object.
    """
    if not isinstance(content, str):
        return "{}"

    cleaned = content.strip()

    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    return cleaned.strip()


def safe_parse_json(content: str, fallback: dict) -> dict:
    try:
        return json.loads(clean_json_response(content))
    except Exception:
        fallback["raw_response"] = content
        return fallback


def get_topic_system_prompt(bucket: str) -> str:
    output_language = get_output_language(bucket)

    return f"""
You are an expert audience insight analyst.

Your task is to analyze YouTube audience discussion topics.

You will receive:
- topic keywords
- representative audience comments
- comment counts

Generate:
1. topic_title
2. interpretation
3. key_finding
4. audience_takeaway

Rules:
- Write the output in {output_language}.
- Be concise and professional.
- Write like a modern analytics dashboard.
- Avoid generic statements such as "General Discussion".
- Focus on audience behavior, content perception, and discussion patterns.
- Do not include usernames.
- Do not expose raw comments.
- Do not wrap JSON in markdown.
- Output ONLY valid JSON.
"""


def build_topic_prompt(row: pd.Series) -> str:
    examples = []

    for col in ["example_1", "example_2", "example_3"]:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()

            if text:
                examples.append(text[:600])

    prompt = f"""
Topic Keywords:
{row["top_keywords"]}

Comment Count:
{row["comment_count"]}

Representative Comments:
{json.dumps(examples, ensure_ascii=False, indent=2)}

Return JSON with exactly these fields:
topic_title, interpretation, key_finding, audience_takeaway
"""
    return prompt


def interpret_topic(row: pd.Series, bucket: str) -> dict:
    prompt = build_topic_prompt(row)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": get_topic_system_prompt(bucket)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )

    content = response.choices[0].message.content

    fallback = {
        "topic_title": "Audience Theme",
        "interpretation": "",
        "key_finding": "",
        "audience_takeaway": "",
    }

    parsed = safe_parse_json(content, fallback=fallback)

    return {
        "topic_title": parsed.get("topic_title") or "Audience Theme",
        "interpretation": parsed.get("interpretation") or "",
        "key_finding": parsed.get("key_finding") or "",
        "audience_takeaway": parsed.get("audience_takeaway") or "",
    }


def generate_ai_topic_insights(run_path: Path):
    reports_dir = run_path / "reports"
    topic_files = list(reports_dir.glob("*_topic_summary.csv"))

    if not topic_files:
        print("⚠️ No topic summary files found.")
        return

    for topic_file in topic_files:
        bucket = topic_file.name.replace("_topic_summary.csv", "")

        print(f"🧠 Generating AI insights for {bucket}...")

        df = pd.read_csv(topic_file)

        if df.empty:
            print(f"⚠️ {bucket}: empty topic summary, skipped.")
            continue

        insights = []

        for _, row in df.iterrows():
            result = interpret_topic(row, bucket)

            insights.append(
                {
                    "bucket": bucket,
                    "cluster_id": row["cluster_id"],
                    "comment_count": row["comment_count"],
                    "top_keywords": row["top_keywords"],
                    "topic_title": result.get("topic_title", ""),
                    "interpretation": result.get("interpretation", ""),
                    "key_finding": result.get("key_finding", ""),
                    "audience_takeaway": result.get("audience_takeaway", ""),
                }
            )

        insights_df = pd.DataFrame(insights)

        output_path = reports_dir / f"{bucket}_topic_insights.csv"

        insights_df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"✅ AI insights saved -> {output_path.name}")


def get_report_system_prompt(bucket: str) -> str:
    output_language = get_output_language(bucket)

    return f"""
You are an expert audience intelligence analyst.

You will receive:
- sentiment distribution
- AI-generated topic insights
- topic sizes

Generate an executive audience report.

Return JSON with exactly these fields:
1. executive_summary
2. key_findings: array of 3 concise findings
3. content_opportunities: array of 3 practical recommendations

Rules:
- Write in {output_language}.
- Be specific and insight-driven.
- Avoid generic wording.
- Do not mention raw comments.
- Do not mention usernames.
- Do not wrap JSON in markdown.
- Output ONLY valid JSON.
"""


def build_overall_report_prompt(
    sentiment_df: pd.DataFrame,
    topic_df: pd.DataFrame,
) -> str:

    sentiment_records = sentiment_df.to_dict(orient="records")

    topic_records = topic_df[
        [
            "topic_title",
            "comment_count",
            "interpretation",
            "key_finding",
            "audience_takeaway",
        ]
    ].to_dict(orient="records")

    prompt = f"""
Sentiment Distribution:
{json.dumps(sentiment_records, ensure_ascii=False, indent=2)}

Topic Insights:
{json.dumps(topic_records, ensure_ascii=False, indent=2)}

Generate the executive audience report JSON.
"""
    return prompt


def generate_overall_report_for_bucket(run_path: Path, bucket: str):
    reports_dir = run_path / "reports"

    sentiment_path = reports_dir / f"{bucket}_sentiment_summary.csv"
    insights_path = reports_dir / f"{bucket}_topic_insights.csv"

    if not sentiment_path.exists() or not insights_path.exists():
        print(f"⚠️ {bucket}: overall report skipped.")
        return

    sentiment_df = pd.read_csv(sentiment_path)
    topic_df = pd.read_csv(insights_path)

    if sentiment_df.empty or topic_df.empty:
        print(f"⚠️ {bucket}: empty report input, skipped.")
        return

    prompt = build_overall_report_prompt(sentiment_df, topic_df)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": get_report_system_prompt(bucket)},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )

    content = response.choices[0].message.content

    fallback = {
        "executive_summary": "",
        "key_findings": [],
        "content_opportunities": [],
    }

    report = safe_parse_json(content, fallback=fallback)

    output_path = reports_dir / f"{bucket}_overall_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Overall AI report saved -> {output_path.name}")


def generate_ai_reports(run_path: Path):
    generate_ai_topic_insights(run_path)

    reports_dir = run_path / "reports"
    insight_files = list(reports_dir.glob("*_topic_insights.csv"))

    for insight_file in insight_files:
        bucket = insight_file.name.replace("_topic_insights.csv", "")
        generate_overall_report_for_bucket(run_path, bucket)