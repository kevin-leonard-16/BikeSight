"""Download, normalize, and load Lyft bike-share trip data into Postgres."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from bikesight.datasets import DATASETS, DatasetConfig

load_dotenv()

DEFAULT_DATA_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
PG_CONN_STR = os.getenv("PG_CONN_STR")


def get_engine() -> Engine:
    if not PG_CONN_STR:
        raise RuntimeError("Set PG_CONN_STR env var before running ingestion.")
    return create_engine(PG_CONN_STR, pool_pre_ping=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BikeSight ingestion runner")
    parser.add_argument("--system", required=True, choices=DATASETS.keys())
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", required=True, type=int, choices=range(1, 13))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and re-load even if raw file already exists.",
    )
    return parser.parse_args()


def build_remote_url(cfg: DatasetConfig, year: int, month: int) -> str:
    yyyymm = f"{year}{month:02d}"
    return f"{cfg.base_url}/{cfg.file_pattern.format(yyyymm=yyyymm)}"


def ensure_raw_path(cfg: DatasetConfig, year: int, month: int) -> Path:
    target_dir = DEFAULT_DATA_DIR / cfg.system_name / str(year) / f"{month:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    if dest.exists() and not overwrite:
        print(f"Raw file already exists: {dest}")
        return dest

    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    return dest


def extract_csv_path(download_path: Path, overwrite: bool = False) -> Path:
    if download_path.suffix == ".zip":
        with ZipFile(download_path) as zf:
            first_member = zf.namelist()[0]
            csv_path = download_path.with_suffix("")
            if csv_path.exists() and not overwrite:
                return csv_path
            print(f"Extracting {first_member} -> {csv_path}")
            with zf.open(first_member) as zipped, open(csv_path, "wb") as out:
                out.write(zipped.read())
            return csv_path
    return download_path


def normalize_frame(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    rename_map = {
        "ride_id": "trip_id",
        "started_at": "start_time",
        "ended_at": "end_time",
        "start_station_id": "start_station_id",
        "end_station_id": "end_station_id",
        "member_casual": "member_casual",
        "user_type": "member_casual",
        "usertype": "member_casual",
        "rideable_type": "rideable_type",
    }

    df = df.rename(columns=rename_map)

    if "trip_id" not in df:
        raise ValueError("Trip ID column missing after normalization.")

    if "member_casual" in df:
        df["member_casual"] = df["member_casual"].replace(
            {"Subscriber": "member", "Customer": "casual", "member": "member", "casual": "casual"}
        )
    else:
        df["member_casual"] = pd.NA

    # Parse datetimes and convert to UTC for warehouse consistency.
    df["start_time"] = _parse_datetime(df.get("start_time"), cfg.timezone)
    df["end_time"] = _parse_datetime(df.get("end_time"), cfg.timezone)

    df = df.dropna(subset=["start_time", "end_time"])

    df["duration_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
    df["month"] = df["start_time"].dt.to_period("M").dt.to_timestamp()
    df["system_name"] = cfg.system_name

    optional_cols = [
        "start_station_id",
        "end_station_id",
        "member_casual",
        "rideable_type",
    ]
    for col in optional_cols:
        if col not in df:
            df[col] = pd.NA

    required = ["trip_id", "start_time", "end_time", "system_name"]
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[
        [
            "trip_id",
            "system_name",
            "start_time",
            "end_time",
            "start_station_id",
            "end_station_id",
            "member_casual",
            "rideable_type",
            "duration_min",
            "month",
        ]
    ]


def _parse_datetime(series: Optional[pd.Series], timezone: str) -> pd.Series:
    if series is None:
        raise ValueError("Datetime column missing from dataset.")
    dt_series = pd.to_datetime(series, errors="coerce")
    if dt_series.dt.tz is None:
        dt_series = dt_series.dt.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    return dt_series.dt.tz_convert("UTC")


def extract_station_frame(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    start_cols = [
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng",
    ]
    end_cols = [
        "end_station_id",
        "end_station_name",
        "end_lat",
        "end_lng",
    ]
    frames = []
    if set(start_cols).issubset(df.columns):
        frames.append(
            df[start_cols]
            .rename(
                columns={
                    "start_station_id": "station_id",
                    "start_station_name": "name",
                    "start_lat": "lat",
                    "start_lng": "lon",
                }
            )
            .assign(system_name=cfg.system_name)
        )
    if set(end_cols).issubset(df.columns):
        frames.append(
            df[end_cols]
            .rename(
                columns={
                    "end_station_id": "station_id",
                    "end_station_name": "name",
                    "end_lat": "lat",
                    "end_lng": "lon",
                }
            )
            .assign(system_name=cfg.system_name)
        )
    if not frames:
        return pd.DataFrame(columns=["station_id", "system_name", "name", "lat", "lon"])
    stations = pd.concat(frames, ignore_index=True)
    stations = stations.dropna(subset=["station_id"]).drop_duplicates(subset=["station_id", "system_name"])
    return stations[["station_id", "system_name", "name", "lat", "lon"]]


def upsert_system(engine: Engine, cfg: DatasetConfig) -> None:
    stmt = text(
        """
        INSERT INTO systems (system_name, display_name, timezone)
        VALUES (:system_name, :display_name, :timezone)
        ON CONFLICT (system_name) DO UPDATE
        SET display_name = EXCLUDED.display_name,
            timezone = EXCLUDED.timezone
        """
    )
    with engine.begin() as conn:
        conn.execute(stmt, cfg.__dict__)


def load_trips(engine: Engine, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with engine.begin() as conn:
        df.to_sql("trips", conn, if_exists="append", index=False, method="multi", chunksize=5000)
    return len(df)


def upsert_stations(engine: Engine, stations_df: pd.DataFrame) -> int:
    if stations_df.empty:
        return 0
    stmt = text(
        """
        INSERT INTO stations (station_id, system_name, name, lat, lon)
        VALUES (:station_id, :system_name, :name, :lat, :lon)
        ON CONFLICT (station_id, system_name) DO UPDATE
        SET name = COALESCE(EXCLUDED.name, stations.name),
            lat = COALESCE(EXCLUDED.lat, stations.lat),
            lon = COALESCE(EXCLUDED.lon, stations.lon),
            updated_at = NOW()
        """
    )
    payload = stations_df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)


def main() -> None:
    args = parse_args()
    cfg = DATASETS[args.system]
    engine = get_engine()

    upsert_system(engine, cfg)

    remote_url = build_remote_url(cfg, args.year, args.month)
    raw_dir = ensure_raw_path(cfg, args.year, args.month)
    raw_zip = raw_dir / Path(remote_url).name
    downloaded = download_file(remote_url, raw_zip, overwrite=args.overwrite)
    csv_path = extract_csv_path(downloaded, overwrite=args.overwrite)

    print(f"Loading CSV {csv_path}")
    df = pd.read_csv(csv_path)
    trips_df = normalize_frame(df.copy(), cfg)
    stations_df = extract_station_frame(df, cfg)

    inserted_stations = upsert_stations(engine, stations_df)
    inserted_trips = load_trips(engine, trips_df)

    print(f"Inserted {inserted_trips} trips and {inserted_stations} stations for {cfg.display_name} {args.year}-{args.month:02d}")


if __name__ == "__main__":
    main()
