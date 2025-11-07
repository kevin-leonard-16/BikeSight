# BikeSight

BikeSight is an MVP analytics app for exploring Lyft-operated bike share systems (Bay Wheels + Citi Bike). It bundles an ingestion script, a Postgres warehouse schema, and a Streamlit dashboard to visualize KPIs and narrative summaries.

## Structure
- `etl/`: Python ingestion + normalization scripts.
- `app/`: Streamlit UI and supporting modules.
- `data/raw/`: Local cache of downloaded CSVs prior to loading into Postgres.
- `docs/`: Design notes, onboarding runbooks, and ERDs.

## Getting Started
### Local Python workflow
1. Create a Python 3.11 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Start Postgres (Docker/local), run `docs/schema.sql`.
4. Ingest a month: `python etl/ingest.py --system baywheels --year 2024 --month 01`.
5. Launch UI: `streamlit run app/dashboard.py`.

### Docker Compose workflow
1. Build services: `docker compose build`.
2. Start Postgres (and optional dashboard): `docker compose up -d db app`.
3. Apply schema inside db container: `docker compose exec db psql -U postgres -d bikesight -f /app/docs/schema.sql`.
4. Run ETL via helper script (wraps `docker compose run`): `./scripts/ingest_month.sh baywheels 2024 01`.
5. Open http://localhost:8501 for the Streamlit dashboard.

See `docs/operations.md` for automation/runbook details.
