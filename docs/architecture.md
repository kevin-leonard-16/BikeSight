# BikeSight Architecture (Phase 1)

## Layers
1. **Ingestion**: Python ETL script downloads Lyft CSVs, stores raw snapshots under `data/raw/{system}/{year}/{month}` and loads normalized data into Postgres via SQLAlchemy.
2. **Warehouse**: Postgres hosts canonical `systems`, `stations`, and `trips` tables. Schemas live inside `docs/schema.sql`.
3. **Analytics App**: Streamlit dashboard queries Postgres, aggregates KPIs, renders Altair charts, and prints text summaries.

## Data Flow
```
Source CSV -> data/raw cache -> Pandas/Polars normalization -> Postgres tables -> Streamlit queries -> Charts + AI summary
```

## Automation Roadmap
- Manual invocation during MVP.
- GitHub Action / cron to run `python etl/ingest.py --run-scheduled` monthly once validated.

## Configuration
Environment variables (store in `.env`):
- `PG_CONN_STR`: SQLAlchemy style connection string.
- `DATA_DIR`: override for raw cache (defaults to `data/raw`).

