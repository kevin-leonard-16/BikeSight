# Operations Guide

## 1. Bring up local stack
```bash
docker compose up -d db
# optional: include app service for dashboard
docker compose up -d app
```
This starts Postgres (and Streamlit when desired). The database is accessible on `localhost:5432` with credentials `postgres / postgres`.

## 2. Apply schema
```bash
docker compose exec db psql -U postgres -d bikesight -f /app/docs/schema.sql
```
(Or run the SQL from your host via any client.)

## 3. Ingest a month of data
Use the helper script which runs the ETL container:
```bash
./scripts/ingest_month.sh baywheels 2024 01
./scripts/ingest_month.sh citibike 2024 01 --overwrite
```
The script mounts `data/raw` so downloads persist between runs.

## 4. Validate load
Run quick row-count checks:
```sql
SELECT system_name, COUNT(*) FROM trips GROUP BY 1;
SELECT COUNT(*) FROM stations;
```

## 5. Launch dashboard
If the `app` service is running, open http://localhost:8501. Otherwise run locally:
```bash
streamlit run app/dashboard.py
```
Ensure `PG_CONN_STR` points to the running Postgres instance.

## 6. Monthly automation (future GitHub Action)
1. Cron triggers workflow on the 5th of each month.
2. Workflow pulls latest repo, builds Docker image, runs `scripts/ingest_month.sh <system> <year> <month>` for each dataset.
3. On success, metrics/logs are pushed to monitoring (GitHub artifacts, Slack, etc.).

## 7. Troubleshooting
- **Schema drift**: compare the downloaded CSV header to expected columns; update `normalize_frame` mapping before rerunning.
- **Duplicate loads**: rerun with `--overwrite` to re-download and reinsert (the trips table uses primary keys to prevent duplicates).
- **Slow queries**: verify indexes (`month`, `member_casual`, `rideable_type`) exist; consider partitioning by month as data grows.
