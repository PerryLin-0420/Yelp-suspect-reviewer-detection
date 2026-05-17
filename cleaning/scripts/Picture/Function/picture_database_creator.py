import duckdb
from pathlib import Path
from loguru import logger
from attrs import define, field


@define
class PictureDatabaseCreator:
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

    def create_label_table(self):
        sequence_sql = "CREATE SEQUENCE IF NOT EXISTS label_id_seq START 1;"
        create_sql = '''
        CREATE TABLE IF NOT EXISTS label (
            label_id   INTEGER PRIMARY KEY DEFAULT nextval('label_id_seq'),
            label_name VARCHAR(50) UNIQUE NOT NULL
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info("[Success] Created label table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create label table: {e}")
            raise

    def create_photo_table(self):
        sequence_sql = "CREATE SEQUENCE IF NOT EXISTS photo_id_seq START 1;"
        create_sql = '''
        CREATE TABLE IF NOT EXISTS photo (
            photo_id      INTEGER PRIMARY KEY DEFAULT nextval('photo_id_seq'),
            photo_raw_id  VARCHAR(22) UNIQUE NOT NULL,
            business_id   INTEGER REFERENCES business(business_id),
            label_id      INTEGER REFERENCES label(label_id),
            caption       TEXT
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info("[Success] Created photo table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create photo table: {e}")
            raise

    def insert_label_data_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        try:
            self.conn.execute(f"""
                INSERT INTO label (label_name)
                SELECT DISTINCT label
                FROM read_json(
                    '{path_str}',
                    format  = 'newline_delimited',
                    columns = {{label: 'VARCHAR'}}
                )
                WHERE label IS NOT NULL AND label != ''
                ON CONFLICT DO NOTHING
            """)

            inserted_count = self.conn.execute("SELECT COUNT(*) FROM label").fetchone()[0]
            logger.info(f"[Success] Inserted {inserted_count} distinct labels into label table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert labels: {e}")
            raise

    def insert_photo_data_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        try:
            raw_count = self.conn.execute(f"""
                SELECT COUNT(*) FROM read_json(
                    '{path_str}',
                    format  = 'newline_delimited',
                    columns = {{photo_id: 'VARCHAR'}}
                )
            """).fetchone()[0]
            logger.info(f"[Info] Raw photo count in JSON: {raw_count}")

            self.conn.execute(f"""
                INSERT INTO photo (photo_raw_id, business_id, label_id, caption)
                WITH raw AS (
                    SELECT
                        photo_id    AS photo_raw_id,
                        business_id AS business_raw_id,
                        NULLIF(label, '')   AS label_name,
                        NULLIF(caption, '') AS caption
                    FROM read_json(
                        '{path_str}',
                        format  = 'newline_delimited',
                        columns = {{
                            photo_id:    'VARCHAR',
                            business_id: 'VARCHAR',
                            label:       'VARCHAR',
                            caption:     'VARCHAR'
                        }}
                    )
                )
                SELECT
                    r.photo_raw_id,
                    b.business_id,
                    l.label_id,
                    r.caption
                FROM raw r
                JOIN business b ON r.business_raw_id = b.business_raw_id
                LEFT JOIN label l ON r.label_name    = l.label_name
                ON CONFLICT DO NOTHING
            """)

            inserted_count = self.conn.execute("SELECT COUNT(*) FROM photo").fetchone()[0]
            dropped = raw_count - inserted_count
            if dropped > 0:
                logger.warning(f"[Warning] {dropped} photos dropped (business not found in DB)")
            logger.info(f"[Success] Inserted {inserted_count} photos into photo table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert photos: {e}")
            raise
