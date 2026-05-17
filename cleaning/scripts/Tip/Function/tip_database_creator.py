import duckdb
from pathlib import Path
from loguru import logger
from attrs import define, field


@define
class TipDatabaseCreator:
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

    def create_tip_table(self):
        sequence_sql = "CREATE SEQUENCE IF NOT EXISTS tip_id_seq START 1;"
        create_sql = '''
        CREATE TABLE IF NOT EXISTS tip (
            tip_id           INTEGER  PRIMARY KEY DEFAULT nextval('tip_id_seq'),
            user_id          INTEGER  NOT NULL REFERENCES "user"(user_id),
            business_id      INTEGER  NOT NULL REFERENCES business(business_id),
            text             TEXT     NOT NULL,
            date             TIMESTAMP NOT NULL,
            compliment_count INTEGER  NOT NULL DEFAULT 0
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info("[Success] Created tip table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create tip table: {e}")
            raise

    def insert_tip_data_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        try:
            raw_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM read_json(
                    '{path_str}',
                    format  = 'newline_delimited',
                    columns = {{user_id: 'VARCHAR'}}
                )
            """).fetchone()[0]
            logger.info(f"[Info] Raw tip count in JSON: {raw_count}")

            self.conn.execute(f"""
                INSERT INTO tip (user_id, business_id, text, date, compliment_count)
                WITH raw AS (
                    SELECT
                        user_id          AS user_raw_id,
                        business_id      AS business_raw_id,
                        text,
                        TRY_CAST(date AS TIMESTAMP) AS date,
                        compliment_count
                    FROM read_json(
                        '{path_str}',
                        format  = 'newline_delimited',
                        columns = {{
                            user_id:          'VARCHAR',
                            business_id:      'VARCHAR',
                            text:             'VARCHAR',
                            date:             'VARCHAR',
                            compliment_count: 'INTEGER'
                        }}
                    )
                    WHERE text IS NOT NULL
                )
                SELECT
                    u.user_id,
                    b.business_id,
                    r.text,
                    r.date,
                    COALESCE(r.compliment_count, 0)
                FROM raw r
                JOIN "user"   u ON r.user_raw_id     = u.user_raw_id
                JOIN business b ON r.business_raw_id = b.business_raw_id
                WHERE r.date IS NOT NULL
                ON CONFLICT DO NOTHING
            """)

            inserted_count = self.conn.execute("SELECT COUNT(*) FROM tip").fetchone()[0]
            dropped = raw_count - inserted_count
            if dropped > 0:
                logger.warning(f"[Warning] {dropped} tips dropped (user or business not found in DB)")
            logger.info(f"[Success] Inserted {inserted_count} tips into tip table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert tips: {e}")
            raise
