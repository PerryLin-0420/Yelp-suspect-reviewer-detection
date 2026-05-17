import duckdb
from pathlib import Path
from loguru import logger
from attrs import define, field

@define
class ReviewDatabaseCreator:
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

    def create_review_table(self):
        sequence_sql = '''
        CREATE SEQUENCE IF NOT EXISTS review_id_seq START 1;
        '''
        create_sql = '''
        CREATE TABLE IF NOT EXISTS review (
            review_id       INTEGER PRIMARY KEY DEFAULT nextval('review_id_seq'),
            review_raw_id   VARCHAR(22) UNIQUE NOT NULL,
            user_id         INTEGER REFERENCES "user"(user_id),
            business_id     INTEGER REFERENCES business(business_id),
            stars           FLOAT,
            useful          INTEGER NOT NULL DEFAULT 0,
            funny           INTEGER NOT NULL DEFAULT 0,
            cool            INTEGER NOT NULL DEFAULT 0,
            text            TEXT,
            date            TIMESTAMP
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created review table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create review table: {e}")
            raise

    def insert_review_data_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        try:
            raw_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM read_json(
                    '{path_str}',
                    format  = 'newline_delimited',
                    columns = {{review_id: 'VARCHAR'}}
                )
            """).fetchone()[0]
            logger.info(f"[Info] Raw review count in JSON: {raw_count}")

            self.conn.execute(f"""
                INSERT INTO review (
                    review_raw_id, user_id, business_id,
                    stars, useful, funny, cool, text, date
                )
                WITH raw AS (
                    SELECT
                        review_id   AS review_raw_id,
                        user_id     AS user_raw_id,
                        business_id AS business_raw_id,
                        stars, useful, funny, cool, text,
                        TRY_CAST(date AS TIMESTAMP) AS date
                    FROM read_json(
                        '{path_str}',
                        format  = 'newline_delimited',
                        columns = {{
                            review_id:   'VARCHAR',
                            user_id:     'VARCHAR',
                            business_id: 'VARCHAR',
                            stars:       'FLOAT',
                            useful:      'INTEGER',
                            funny:       'INTEGER',
                            cool:        'INTEGER',
                            text:        'VARCHAR',
                            date:        'VARCHAR'
                        }}
                    )
                )
                SELECT
                    r.review_raw_id,
                    u.user_id,
                    b.business_id,
                    r.stars, r.useful, r.funny, r.cool, r.text, r.date
                FROM raw r
                JOIN "user"   u ON r.user_raw_id     = u.user_raw_id
                JOIN business b ON r.business_raw_id = b.business_raw_id
                ON CONFLICT DO NOTHING
            """)

            inserted_count = self.conn.execute("SELECT COUNT(*) FROM review").fetchone()[0]
            dropped = raw_count - inserted_count
            if dropped > 0:
                logger.warning(f"[Warning] {dropped} reviews dropped (user or business not found in DB)")
            logger.info(f"[Success] Inserted {inserted_count} reviews into review table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert reviews: {e}")
            raise
