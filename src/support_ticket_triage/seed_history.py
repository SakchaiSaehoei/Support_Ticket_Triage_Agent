"""Prepare sample customer records and earlier messages without calling any API."""

import argparse
import json
from pathlib import Path

from .database import DEFAULT_DATABASE_PATH, DEFAULT_FIXTURE_DIRECTORY, SQLiteStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database path."
    )
    args = parser.parse_args()
    store = SQLiteStore(args.database)
    counts = store.seed_samples(
        DEFAULT_FIXTURE_DIRECTORY / "customer_accounts.json",
        DEFAULT_FIXTURE_DIRECTORY / "sample_tickets.json",
    )
    print(json.dumps({"inserted": counts}, indent=2))


if __name__ == "__main__":
    main()
