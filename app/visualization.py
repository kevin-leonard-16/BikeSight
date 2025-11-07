"""Chart helpers for the Streamlit dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd


def hourly_histogram(df: pd.DataFrame, title: str) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame({"hour": [], "trips": []}))
    data = df.groupby("hour", as_index=False).size()
    data = data.rename(columns={"size": "trips"})
    return (
        alt.Chart(data)
        .mark_bar(color="#1f77b4")
        .encode(x=alt.X("hour:O", title="Hour of day"), y=alt.Y("trips:Q", title="Trips"))
        .properties(height=240, title=title)
    )


def share_chart(df: pd.DataFrame, column: str, title: str) -> alt.Chart:
    if df.empty or column not in df:
        chart_df = pd.DataFrame({column: [], "trips": []})
    else:
        chart_df = df[column].value_counts().rename_axis(column).reset_index(name="trips")
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(x=alt.X(column, title=""), y=alt.Y("trips", title="Trips"))
        .properties(height=200, title=title)
    )
