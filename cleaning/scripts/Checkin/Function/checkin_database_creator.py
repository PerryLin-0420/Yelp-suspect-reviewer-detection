import duckdb
from pathlib import Path
from loguru import logger
from attrs import define, field


@define
class CheckinDatabaseCreator:
    conn: any = field(init=False, default=None)

    def db_connection(self, output_path: Path = None):
        try:
            self.conn = duckdb.connect(output_path)
            logger.info(f"[Success] Connected to database: {output_path}")
            return self.conn
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to database: {e}")
            raise

    def db_close(self):
        try:
            if self.conn:
                self.conn.close()
                logger.info(f"[Success] Database connection closed")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to close database connection: {e}")
            raise

    def db_drop(self, table_name=""):
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            logger.info(f"[Success] Dropped {table_name} table if it existed")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to drop {table_name} table: {e}")
            raise

    def db_begin(self):
        self.conn.execute("BEGIN")

    def db_commit(self):
        self.conn.execute("COMMIT")

    def db_rollback(self):
        try:
            self.conn.execute("ROLLBACK")
        except Exception:
            pass

    def create_checkin_table(self):
        sequence_sql = "CREATE OR REPLACE SEQUENCE checkin_id_seq START 1;"
        create_sql = '''
        CREATE TABLE IF NOT EXISTS checkin (
            checkin_id   INTEGER   PRIMARY KEY DEFAULT nextval('checkin_id_seq'),
            business_id  INTEGER   NOT NULL REFERENCES business(business_id),
            checkin_time TIMESTAMP NOT NULL
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info("[Success] Created checkin table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create checkin table: {e}")
            raise

    def insert_checkin_data_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        try:
            expanded_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM (
                    SELECT TRIM(UNNEST(string_split(date, ','))) AS ts_str
                    FROM read_json(
                        '{path_str}',
                        format  = 'newline_delimited',
                        columns = {{
                            business_id: 'VARCHAR',
                            date:        'VARCHAR'
                        }}
                    )
                    WHERE date IS NOT NULL
                ) WHERE ts_str <> ''
            """).fetchone()[0]
            logger.info(f"[Info] Total checkin events in JSON: {expanded_count}")

            self.conn.execute(f"""
                INSERT INTO checkin (business_id, checkin_time)
                WITH raw AS (
                    SELECT
                        business_id                              AS business_raw_id,
                        TRIM(UNNEST(string_split(date, ',')))   AS ts_str
                    FROM read_json(
                        '{path_str}',
                        format  = 'newline_delimited',
                        columns = {{
                            business_id: 'VARCHAR',
                            date:        'VARCHAR'
                        }}
                    )
                    WHERE date IS NOT NULL
                ),
                parsed AS (
                    SELECT
                        business_raw_id,
                        TRY_CAST(ts_str AS TIMESTAMP) AS checkin_time
                    FROM raw
                    WHERE ts_str <> ''
                )
                SELECT
                    b.business_id,
                    p.checkin_time
                FROM parsed p
                JOIN business b ON p.business_raw_id = b.business_raw_id
                WHERE p.checkin_time IS NOT NULL
            """)

            inserted_count = self.conn.execute("SELECT COUNT(*) FROM checkin").fetchone()[0]
            dropped = expanded_count - inserted_count
            if dropped > 0:
                logger.warning(f"[Warning] {dropped} checkin events dropped (business not found in DB or invalid timestamp)")
            logger.info(f"[Success] Inserted {inserted_count} checkin events into checkin table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert checkins: {e}")
            raise
