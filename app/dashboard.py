"""Streamlit dashboard for BikeSight."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from app.visualization import hourly_histogram, share_chart
from bikesight.datasets import DATASETS, get_dataset

load_dotenv()

st.set_page_config(page_title="BikeSight", layout="wide")


def _get_conn_str() -> str | None:
    return os.getenv("PG_CONN_STR")


@st.cache_resource(show_spinner=False)
def get_engine():  # type: ignore[override]
    conn_str = _get_conn_str()
    if not conn_str:
        raise RuntimeError("PG_CONN_STR env var missing.")
    return create_engine(conn_str, pool_pre_ping=True)


@st.cache_data(ttl=600)
def fetch_trips(system: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    engine = get_engine()
    query = text(
        """
        SELECT trip_id, system_name, start_time, end_time, start_station_id, end_station_id,
               member_casual, rideable_type, duration_min
        FROM trips
        WHERE system_name = :system
          AND start_time >= :start_ts AND start_time < :end_ts
        ORDER BY start_time
        """
    )
    with engine.begin() as conn:
        df = pd.read_sql_query(query, conn, params={"system": system, "start_ts": start_ts, "end_ts": end_ts})
    return df


def build_summary(metrics: dict, cfg_display: str, start_label: str, end_label: str) -> str:
    total = metrics.get("total", 0)
    if total == 0:
        return f"No rides found for {cfg_display} between {start_label} and {end_label}."
    peak_hour = metrics.get("peak_hour")
    member_share = metrics.get("member_share", 0)
    avg_duration = metrics.get("avg_duration", 0)
    ebike_share = metrics.get("ebike_share", 0)
    sentences = [
        f"Between {start_label} and {end_label}, riders completed {total:,} trips on {cfg_display}."
    ]
    sentences.append(f"Average ride lasted {avg_duration:.1f} minutes with members making up {member_share:.0%} of trips.")
    if peak_hour is not None:
        sentences.append(f"Demand peaked around {peak_hour}:00, and e-bikes accounted for {ebike_share:.0%} of rides.")
    return " ".join(sentences)


def compute_metrics(df: pd.DataFrame, cfg_timezone: str) -> dict:
    if df.empty:
        return {"total": 0}
    metrics = {
        "total": len(df),
        "avg_duration": df["duration_min"].mean(),
    }
    members = df["member_casual"].value_counts(normalize=True)
    metrics["member_share"] = members.get("member", 0.0)
    ebikes = df["rideable_type"].value_counts(normalize=True)
    metrics["ebike_share"] = ebikes.get("electric_bike", 0.0)

    df_local = df.copy()
    df_local["start_time"] = pd.to_datetime(df_local["start_time"], utc=True).dt.tz_convert(cfg_timezone)
    df_local["hour"] = df_local["start_time"].dt.hour
    metrics["peak_hour"] = df_local["hour"].mode().iat[0] if not df_local["hour"].empty else None
    metrics["hourly"] = df_local[["hour"]]
    metrics["member_casual"] = df[["member_casual"]]
    metrics["rideable_type"] = df[["rideable_type"]]
    return metrics


def _to_utc(day: date, cfg_timezone: str, end_of_day: bool = False) -> datetime:
    base_time = time.max if end_of_day else time.min
    local_dt = datetime.combine(day, base_time)
    localized = local_dt.replace(tzinfo=ZoneInfo(cfg_timezone))
    if end_of_day:
        localized = localized.replace(hour=23, minute=59, second=59)
    return localized.astimezone(ZoneInfo("UTC"))


def sidebar_controls() -> tuple[str, date, date]:
    st.sidebar.header("Filters")
    system = st.sidebar.selectbox("System", options=list(DATASETS.keys()), format_func=lambda key: DATASETS[key].display_name)
    today = date.today()
    default_start = today - timedelta(days=30)
    date_range = st.sidebar.date_input("Date range", value=(default_start, today))
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    return system, start_date, end_date


def main() -> None:
    st.title("BikeSight Dashboard")
    conn_str = _get_conn_str()
    if not conn_str:
        st.warning("Set PG_CONN_STR in your environment to use the dashboard.")
        st.stop()

    system, start_day, end_day = sidebar_controls()
    cfg = get_dataset(system)
    start_utc = _to_utc(start_day, cfg.timezone)
    end_utc = _to_utc(end_day, cfg.timezone, end_of_day=True)

    with st.spinner("Loading trips..."):
        trips_df = fetch_trips(system, start_utc, end_utc)

    metrics = compute_metrics(trips_df, cfg.timezone)
    summary = build_summary(
        metrics,
        cfg.display_name,
        start_day.strftime("%b %d, %Y"),
        end_day.strftime("%b %d, %Y"),
    )

    st.subheader("AI Summary (templated)")
    st.info(summary)

    st.subheader("Key Metrics")
    cols = st.columns(4)
    cols[0].metric("Total Rides", f"{metrics.get('total', 0):,}")
    cols[1].metric("Avg Duration (min)", f"{metrics.get('avg_duration', 0):.1f}")
    cols[2].metric("Member Share", f"{metrics.get('member_share', 0)*100:.0f}%")
    cols[3].metric("E-bike Share", f"{metrics.get('ebike_share', 0)*100:.0f}%")

    st.subheader("Visualizations")
    chart_cols = st.columns(2)
    hourly_df = metrics.get("hourly", pd.DataFrame(columns=["hour"]))
    chart_cols[0].altair_chart(hourly_histogram(hourly_df, "Trips by hour"), use_container_width=True)
    member_df = metrics.get("member_casual", pd.DataFrame(columns=["member_casual"]))
    chart_cols[1].altair_chart(share_chart(member_df, "member_casual", "Member vs casual"), use_container_width=True)

    chart_cols_2 = st.columns(2)
    rideable_df = metrics.get("rideable_type", pd.DataFrame(columns=["rideable_type"]))
    chart_cols_2[0].altair_chart(share_chart(rideable_df, "rideable_type", "E-bike vs classic"), use_container_width=True)

    st.subheader("Raw Data Preview")
    st.dataframe(trips_df.head(100))


if __name__ == "__main__":
    main()
