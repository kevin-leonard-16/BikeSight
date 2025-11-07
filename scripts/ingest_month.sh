#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <system> <year> <month> [--overwrite]"
  exit 1
fi

SYSTEM=$1
YEAR=$2
MONTH=$3
shift 3

if ! command -v docker &>/dev/null; then
  echo "Docker is required to run this script." >&2
  exit 1
fi

COMPOSE_PROFILES=etl docker compose run --rm --entrypoint "" etl \
  python etl/ingest.py --system "$SYSTEM" --year "$YEAR" --month "$MONTH" "$@"
