from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


def _read_schema_sql() -> str:
    """Load `schema.sql` next to this module as UTF-8 text."""
    schema_path = Path(__file__).with_name("schema.sql")
    return schema_path.read_text(encoding="utf-8")


def main() -> None:
    """Apply `schema.sql` to the database from `DATABASE_URL`."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set. Create a .env file (see .env.example).")

    engine = create_engine(db_url, future=True)
    schema_sql = _read_schema_sql()

    # Execute schema as raw SQL (idempotent via IF NOT EXISTS).
    with engine.begin() as conn:
        for stmt in [s.strip() for s in schema_sql.split(";") if s.strip()]:
            conn.execute(text(stmt))

    log.info("DB initialized: schema applied successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
