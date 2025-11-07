"""Dataset metadata shared by the ETL and dashboard layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    system_name: str
    display_name: str
    base_url: str
    file_pattern: str
    timezone: str


DATASETS: dict[str, DatasetConfig] = {
    "baywheels": DatasetConfig(
        system_name="baywheels",
        display_name="Bay Wheels",
        base_url="https://s3.amazonaws.com/baywheels-data",
        file_pattern="{yyyymm}-baywheels-tripdata.csv.zip",
        timezone="America/Los_Angeles",
    ),
    "citibike": DatasetConfig(
        system_name="citibike",
        display_name="Citi Bike",
        base_url="https://s3.amazonaws.com/tripdata",
        file_pattern="{yyyymm}-citibike-tripdata.csv.zip",
        timezone="America/New_York",
    ),
}


def get_dataset(system_name: str) -> DatasetConfig:
    try:
        return DATASETS[system_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported system: {system_name}") from exc
