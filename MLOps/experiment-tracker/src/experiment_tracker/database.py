import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List

class ExperimentDB:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Creates a connection and ensures that it is closed"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _init_db(self):
        queries = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                source_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(id)
                    ON DELETE CASCADE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES runs(id)
                    ON DELETE CASCADE,
                UNIQUE(run_id, key)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES runs(id)
                    ON DELETE CASCADE
            );
            """
        ]

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for query in queries:
                cursor.execute(query)

    def create_experiment(self, name: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO experiments (name, created_at) VALUES (?, ?);",
                    (name, self._now())
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                cursor.execute("SELECT id FROM experiments WHERE name = ?;", (name,))
                row = cursor.fetchone()
                return row["id"]

    def create_run(self, experiment_id: int, run_name: str, source_type: str) -> str:
        run_id = uuid.uuid4().hex
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (id, experiment_id, name, status, source_type, start_time)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (run_id, experiment_id, run_name, "RUNNING", source_type, self._now())
            )
        return run_id

    def update_run_status(self, run_id: str, status: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET status = ?, end_time = ? WHERE id = ?;",
                (status, self._now(), run_id)
            )

    def log_parameter(self, run_id: str, key: str, value: Any):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO params (run_id, key, value)
                VALUES (?, ?, ?);
                """,
                (run_id, key, str(value))
            )

    def log_metric(self, run_id: str, key: str, value: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (run_id, key, value, timestamp)
                VALUES (?, ?, ?, ?);
                """,
                (run_id, key, value, self._now())
            )

    def get_all_experiments(self) -> List[sqlite3.Row]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, created_at FROM experiments ORDER BY created_at DESC;")
            return cursor.fetchall()

    def get_experiment_runs_summary(self, experiment_name: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM experiments WHERE name = ?;", (experiment_name,))
            exp_row = cursor.fetchone()
            if not exp_row:
                return []

            exp_id = exp_row["id"]

            cursor.execute(
                """
                SELECT id, name, status, source_type, start_time, end_time
                FROM runs
                WHERE experiment_id = ?
                ORDER BY start_time DESC;
                """,
                (exp_id,)
            )
            runs = cursor.fetchall()

            summary = []
            for run in runs:
                run_id = run["id"]
                run_dict = {
                    "ID": run_id[:8],
                    "Name": run["name"],
                    "Status": run["status"],
                    "Source": run["source_type"],
                    "Started": run["start_time"].split("T")[1][:8],
                }

                cursor.execute("SELECT key, value FROM params WHERE run_id = ?;", (run_id,))
                for param in cursor.fetchall():
                    run_dict[f"p:{param['key']}"] = param["value"]

                cursor.execute("SELECT key, value FROM metrics WHERE run_id = ?;", (run_id,))
                for metric in cursor.fetchall():
                    run_dict[f"m:{metric['key']}"] = round(metric["value"], 4)

                summary.append(run_dict)

            return summary


if __name__ == "__main__":
    db = ExperimentDB("test.db")

    with db._get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table';
        """)

        print([row["name"] for row in cursor.fetchall()])

