# Ingestion Runbook

1. Export environment variables (see `.env.example`).
2. Apply the schema in `docs/schema.sql` to your Postgres instance.
3. Run the ingestion script for a system + month, e.g.
   ```bash
   python etl/ingest.py --system baywheels --year 2024 --month 1
   ```
4. Verify row counts in Postgres before moving on to dashboard work.

Flags:
- `--overwrite` re-downloads and re-extracts raw files if you need a clean run.

The script automatically inserts/upserts entries into `systems`, `stations`, and `trips`.
