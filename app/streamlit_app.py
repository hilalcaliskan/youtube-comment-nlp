import base64
import html
import json
import re
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="InsightTube",
    layout="wide",
)


st.markdown(
    """
<style>
.main-title {
    font-size: 52px;
    font-weight: 850;
    margin-bottom: 6px;
}

.subtitle {
    font-size: 19px;
    color: #667085;
    margin-bottom: 33px;
}

.video-title {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 12px;
}

.stat-card {
    padding: 24px;
    border-radius: 22px;
    border: 1px solid #EAECF0;
    background: #FFFFFF;
    box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    min-height: 130px;
}

.stat-label {
    color: #667085;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700;
}

.stat-value {
    font-size: 34px;
    font-weight: 850;
    margin-top: 12px;
}

.summary-box {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(135deg, #111827, #26337A);
    color: white;
    margin-top: 28px;
    margin-bottom: 34px;
}

.finding-card {
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #EAECF0;
    background: white;
    box-shadow: 0 3px 12px rgba(16,24,40,0.05);
    min-height: 190px;
}

.topic-card {
    padding: 26px;
    border-radius: 22px;
    border: 1px solid #EAECF0;
    background: white;
    box-shadow: 0 4px 14px rgba(16,24,40,0.05);
    min-height: 300px;
    margin-bottom: 20px;
}

.small-label {
    color: #667085;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 750;
}

.insight-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 12px;
}

.insight-text {
    color: #344054;
    font-size: 15px;
    line-height: 1.65;
}

.takeaway-box {
    margin-top: 18px;
    padding: 15px;
    border-radius: 14px;
    background: #F8FAFC;
    color: #344054;
    font-size: 14px;
    line-height: 1.55;
}

.app-header {
    display: flex;
    align-items: flex-end;
    gap: 24px;
    margin-bottom: 30px;
}

.app-logo {
    width: 220px;
    max-width: 220px;
    height: auto;
}

.header-text-block {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding-bottom: 18px;
}

@media (max-width: 768px) {
    .app-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
    }

    .app-logo {
        width: 100px;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(r"```json|```", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("nan", "").strip()

    return text


def safe_html(text: str) -> str:
    return html.escape(clean_text(text))




def image_to_base64(image_path: Path) -> str:
    """Convert local image file to base64 for HTML display."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def render_app_header():
    """Render InsightTube header with optional local logo."""
    base_dir = Path(__file__).resolve().parent.parent
    logo_path = base_dir / "assets" / "insighttube_logo.png"

    if logo_path.exists():
        logo_base64 = image_to_base64(logo_path)

        st.markdown(
            f"""
<div class="app-header">
    <img class="app-logo" src="data:image/png;base64,{logo_base64}" alt="InsightTube logo">
    <div class="header-text-block">
        <div class="main-title">InsightTube</div>
        <div class="subtitle">An AI-Powered YouTube Audience Analyzer.</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="main-title">InsightTube</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="subtitle">An AI-powered multilingual YouTube audience analyzer.</div>',
            unsafe_allow_html=True,
        )


def format_number(value) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def get_latest_run() -> Path | None:
    runs_path = Path("runs")
    if not runs_path.exists():
        return None

    runs = [p for p in runs_path.iterdir() if p.is_dir()]
    if not runs:
        return None

    return sorted(runs, reverse=True)[0]


def load_meta(run_path: Path) -> dict:
    meta_path = run_path / "meta.json"

    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_dominant_bucket(reports_dir: Path) -> str | None:
    sentiment_files = list(reports_dir.glob("*_sentiment_summary.csv"))

    if not sentiment_files:
        return None

    bucket_counts = {}

    for file in sentiment_files:
        bucket = file.name.replace("_sentiment_summary.csv", "")
        df = pd.read_csv(file)

        if "count" in df.columns:
            bucket_counts[bucket] = int(df["count"].sum())

    if not bucket_counts:
        return None

    return max(bucket_counts, key=bucket_counts.get)


def get_ratio(sentiment_df: pd.DataFrame, label: str) -> float:
    row = sentiment_df[sentiment_df["sentiment_label"] == label]
    if row.empty:
        return 0.0
    return float(row["ratio"].iloc[0])


def show_video_overview(meta: dict, analyzed_comments: int, bucket: str):
    st.subheader("🎬 Video Overview")

    col_img, col_info = st.columns([1.1, 2])

    with col_img:
        if meta.get("thumbnail_url"):
            st.image(meta["thumbnail_url"], use_container_width=True)

    with col_info:
        st.markdown(
            f"""
<div class="video-title">{safe_html(meta.get("title", "Unknown Video"))}</div>
<div style="color:#667085; font-size:16px;">
Channel: {safe_html(meta.get("channel_title", "-"))}
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        stats = [
            ("Views", format_number(meta.get("view_count", 0))),
            ("Likes", format_number(meta.get("like_count", 0))),
            ("Total Comments", format_number(meta.get("comment_count", 0))),
            ("Analyzed", format_number(analyzed_comments)),
            ("Main Language", bucket.upper()),
            ("Video ID", safe_html(meta.get("video_id", "-"))),
        ]

        for col, (label, value) in zip([c1, c2, c3, c4, c5, c6], stats):
            with col:
                st.markdown(
                    f"""
<div class="stat-card">
    <div class="stat-label">{label}</div>
    <div class="stat-value">{value}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

def show_sentiment_section(sentiment_df: pd.DataFrame):
    st.subheader("📊 Audience Sentiment")

    left, right = st.columns([1.25, 1])

    chart_df = sentiment_df.copy()
    chart_df["sentiment"] = chart_df["sentiment_label"].str.title()

    with left:
        fig = px.pie(
            chart_df,
            names="sentiment",
            values="count",
            hole=0.55,
            custom_data=["count", "ratio"],
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Comments: %{customdata[0]}<br>Share: %{percent}<extra></extra>",
        )

        fig.update_layout(
            height=520,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        for _, row in chart_df.iterrows():
            st.markdown(
                f"""
<div class="stat-card" style="margin-bottom:16px;">
    <div class="stat-label">{safe_html(row["sentiment"])}</div>
    <div class="stat-value">{float(row["ratio"]):.1%}</div>
    <div style="color:#667085;">{int(row["count"]):,} comments</div>
</div>
""",
                unsafe_allow_html=True,
            )


def show_key_findings(overall_report: dict):
    st.subheader("✨ Key Findings")

    findings = overall_report.get("key_findings", [])

    if not findings:
        st.info("No key findings available.")
        return

    findings = [clean_text(x) for x in findings if clean_text(x)]

    if not findings:
        st.info("No key findings available.")
        return

    cols = st.columns(min(3, len(findings)))

    for idx, finding in enumerate(findings[:3]):
        with cols[idx]:
            st.markdown(
                f"""
<div class="finding-card">
    <div class="small-label">Finding {idx + 1}</div>
    <p class="insight-text">{safe_html(finding)}</p>
</div>
""",
                unsafe_allow_html=True,
            )


def show_topic_distribution(topic_df: pd.DataFrame):
    st.subheader("📈 Topic Distribution")

    chart_df = topic_df.copy()
    chart_df["Topic Title"] = chart_df["topic_title"].apply(clean_text)
    chart_df["Comment Count"] = chart_df["comment_count"]

    chart_df = chart_df.sort_values("Comment Count", ascending=True)

    fig = px.bar(
        chart_df,
        x="Comment Count",
        y="Topic Title",
        orientation="h",
        custom_data=["Comment Count"],
    )

    fig.update_traces(
        hovertemplate="Comment Count: %{customdata[0]}<extra></extra>",
    )

    fig.update_layout(
        height=520,
        xaxis_title="Comment Count",
        yaxis_title="Topic Title",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)


def show_topic_cards(topic_df: pd.DataFrame):
    st.subheader("💬 Main Conversation Themes")

    topic_df = topic_df.sort_values("comment_count", ascending=False).reset_index(drop=True)

    cols = st.columns(2)

    for idx, row in topic_df.iterrows():
        title = safe_html(row.get("topic_title", f"Topic {idx + 1}"))
        interpretation = safe_html(row.get("interpretation", ""))
        takeaway = safe_html(row.get("audience_takeaway", ""))

        with cols[idx % 2]:
            st.markdown(
                f"""
<div class="topic-card">
    <div class="small-label">Topic {idx + 1}</div>
    <div class="insight-title">{title}</div>
    <p><b>{int(row["comment_count"]):,} comments</b></p>
    <p class="insight-text">{interpretation}</p>
    <div class="takeaway-box">
        <b>Audience Insight</b><br>
        {takeaway}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )


render_app_header()

youtube_url = st.text_input("Enter YouTube Video URL")


if st.button("Analyze Video"):
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    st.info("Running analysis...")

    process = subprocess.run(
        ["python", "src/run_pipeline.py"],
        input=youtube_url,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        st.error("Analysis failed.")

        with st.expander("Show error details"):
            st.text(process.stderr)

        st.stop()

    st.success("Analysis completed!")

    latest_run = get_latest_run()

    if latest_run is None:
        st.error("No run folder found.")
        st.stop()

    reports_dir = latest_run / "reports"

    dominant_bucket = find_dominant_bucket(reports_dir)

    if dominant_bucket is None:
        st.warning("No sentiment report found.")
        st.stop()

    sentiment_path = reports_dir / f"{dominant_bucket}_sentiment_summary.csv"
    topic_path = reports_dir / f"{dominant_bucket}_topic_insights.csv"
    overall_report_path = reports_dir / f"{dominant_bucket}_overall_report.json"

    if not sentiment_path.exists() or not topic_path.exists():
        st.warning("Required report files missing.")
        st.stop()

    sentiment_df = pd.read_csv(sentiment_path)
    topic_df = pd.read_csv(topic_path)

    overall_report = {}

    if overall_report_path.exists():
        with open(overall_report_path, "r", encoding="utf-8") as f:
            overall_report = json.load(f)

    meta = load_meta(latest_run)
    total_comments = int(sentiment_df["count"].sum())

    show_video_overview(
        meta=meta,
        analyzed_comments=total_comments,
        bucket=dominant_bucket,
    )

    executive_summary = clean_text(overall_report.get("executive_summary", ""))

    st.markdown(
        f"""
<div class="summary-box">
    <div class="small-label">Executive Summary</div>
    <h3>Audience Summary</h3>
    <p>{safe_html(executive_summary)}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    show_sentiment_section(sentiment_df)
    show_key_findings(overall_report)
    show_topic_distribution(topic_df)
    show_topic_cards(topic_df)

    with st.expander("Show pipeline logs"):
        st.text(process.stdout)

    st.caption(f"Run ID: {latest_run.name}")