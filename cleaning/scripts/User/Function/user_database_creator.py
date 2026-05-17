import duckdb
import pandas as pd
from loguru import logger
from pathlib import Path
from attrs import define, field

@define
class UserDatabaseCreator:
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

    def create_user_table(self):
        sequence_sql = '''
        CREATE SEQUENCE IF NOT EXISTS user_id_seq START 1;
        '''
        create_sql = '''
        CREATE TABLE IF NOT EXISTS user (
            user_id         INTEGER PRIMARY KEY DEFAULT nextval('user_id_seq'),
            user_raw_id     VARCHAR(22) UNIQUE NOT NULL,
            name            VARCHAR(100) NOT NULL,
            yelping_since   DATETIME NOT NULL,
            review_count    INTEGER NOT NULL DEFAULT 0,
            fans            INTEGER NOT NULL DEFAULT 0,
            average_stars   DECIMAL(3,2),
            useful          INTEGER NOT NULL DEFAULT 0,
            funny           INTEGER NOT NULL DEFAULT 0,
            cool            INTEGER NOT NULL DEFAULT 0,
            compliment_hot      INTEGER NOT NULL DEFAULT 0,
            compliment_more     INTEGER NOT NULL DEFAULT 0,
            compliment_profile  INTEGER NOT NULL DEFAULT 0,
            compliment_cute     INTEGER NOT NULL DEFAULT 0,
            compliment_list     INTEGER NOT NULL DEFAULT 0,
            compliment_note     INTEGER NOT NULL DEFAULT 0,
            compliment_plain    INTEGER NOT NULL DEFAULT 0,
            compliment_cool     INTEGER NOT NULL DEFAULT 0,
            compliment_funny    INTEGER NOT NULL DEFAULT 0,
            compliment_writer   INTEGER NOT NULL DEFAULT 0,
            compliment_photos   INTEGER NOT NULL DEFAULT 0
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created user table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create user table: {e}")
            raise

    def insert_user_data(self, data: list[tuple]):
        try:
            columns = [
                'user_raw_id', 'name', 'yelping_since', 'review_count', 'fans', 'average_stars',
                'useful', 'funny', 'cool',
                'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
                'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
                'compliment_funny', 'compliment_writer', 'compliment_photos',
            ]
            df = pd.DataFrame(data, columns=columns)
            df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='raise')
            self.conn.register('_user_batch', df)
            try:
                self.conn.execute("""
                    INSERT INTO user (
                        user_raw_id, name, yelping_since, review_count, fans, average_stars,
                        useful, funny, cool,
                        compliment_hot, compliment_more, compliment_profile, compliment_cute,
                        compliment_list, compliment_note, compliment_plain, compliment_cool,
                        compliment_funny, compliment_writer, compliment_photos
                    )
                    SELECT * FROM _user_batch
                    ON CONFLICT DO NOTHING
                """)
            finally:
                self.conn.unregister('_user_batch')
            logger.info(f"[Success] Inserted {len(df)} rows into user table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to insert data into user table: {e}")
            raise

    def get_user_id_map(self, raw_ids: list[str]) -> dict:
        if not raw_ids:
            return {}
        placeholders = ",".join("?" * len(raw_ids))
        rows = self.conn.execute(
            f"SELECT user_raw_id, user_id FROM user WHERE user_raw_id IN ({placeholders})",
            raw_ids
        ).fetchall()
        return {raw_id: user_id for raw_id, user_id in rows}

    def get_full_user_id_map(self) -> dict:
        rows = self.conn.execute("SELECT user_raw_id, user_id FROM user").fetchall()
        logger.info(f"[Success] Loaded full user ID map: {len(rows)} entries")
        return {raw_id: user_id for raw_id, user_id in rows}

    def db_begin(self):
        self.conn.execute("BEGIN")

    def db_commit(self):
        self.conn.execute("COMMIT")

    def db_rollback(self):
        try:
            self.conn.execute("ROLLBACK")
        except Exception:
            pass

    def create_user_friends_table(self):
        create_sql = '''
        CREATE TABLE IF NOT EXISTS user_friends (
            user_id   INTEGER NOT NULL,
            friend_id INTEGER NOT NULL,
            PRIMARY KEY (user_id, friend_id),
            FOREIGN KEY (user_id)   REFERENCES user(user_id),
            FOREIGN KEY (friend_id) REFERENCES user(user_id),
            CHECK (user_id < friend_id)
        );
        '''
        try:
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created user_friends table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create user_friends table: {e}")
            raise

    def insert_user_friends(self, data: list[tuple]):
        # Normalise to (min, max); ON CONFLICT DO NOTHING handles cross-chunk dupes
        try:
            rows = [(min(a, b), max(a, b)) for a, b in data if a != b]
            self.conn.executemany(
                "INSERT INTO user_friends (user_id, friend_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
                rows
            )
            logger.info(f"[Success] Inserted {len(rows)} rows into user_friends table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to insert data into user_friends table: {e}")
            raise

    def insert_user_friends_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        sql = f"""
            INSERT INTO user_friends (user_id, friend_id)
            WITH raw AS (
                SELECT user_id AS user_raw_id,
                       trim(unnest(string_split(friends, ','))) AS friend_raw_id
                FROM read_json(
                    '{path_str}',
                    format      = 'newline_delimited',
                    columns     = {{user_id: 'VARCHAR', friends: 'VARCHAR'}}
                )
                WHERE friends IS NOT NULL
                  AND friends != 'None'
                  AND friends != ''
            ),
            mapped AS (
                SELECT LEAST(u1.user_id, u2.user_id)    AS uid,
                       GREATEST(u1.user_id, u2.user_id) AS fid
                FROM raw r
                JOIN "user" u1 ON r.user_raw_id    = u1.user_raw_id
                JOIN "user" u2 ON r.friend_raw_id  = u2.user_raw_id
                WHERE u1.user_id != u2.user_id
            )
            SELECT DISTINCT uid, fid FROM mapped
            ON CONFLICT DO NOTHING
        """
        try:
            self.conn.execute(sql)
            count = self.conn.execute("SELECT COUNT(*) FROM user_friends").fetchone()[0]
            logger.info(f"[Success] Inserted {count} rows into user_friends table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert user_friends: {e}")
            raise

    def create_user_elite_table(self):
        sql = '''
        CREATE TABLE IF NOT EXISTS user_elite (
            user_id INTEGER NOT NULL,
            year    SMALLINT NOT NULL,
            PRIMARY KEY (user_id, year),
            FOREIGN KEY (user_id) REFERENCES user(user_id)
        );
        '''
        try:
            self.conn.execute(sql)
            logger.info(f"[Success] Created user_elite table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create user_elite table: {e}")
            raise

    def insert_user_elite(self, data: list[tuple]):
        try:
            self.conn.executemany(
                "INSERT INTO user_elite (user_id, year) VALUES (?, ?) ON CONFLICT DO NOTHING",
                data
            )
            logger.info(f"[Success] Inserted {len(data)} rows into user_elite table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to insert data into user_elite table: {e}")
            raise

    def insert_user_elite_bulk(self, input_path: Path):
        path_str = str(input_path).replace('\\', '/')
        sql = f"""
            INSERT INTO user_elite (user_id, year)
            WITH raw AS (
                SELECT user_id AS user_raw_id,
                       trim(unnest(string_split(elite, ','))) AS year_str
                FROM read_json(
                    '{path_str}',
                    format  = 'newline_delimited',
                    columns = {{user_id: 'VARCHAR', elite: 'VARCHAR'}}
                )
                WHERE elite IS NOT NULL
                  AND elite != 'None'
                  AND elite != ''
            ),
            mapped AS (
                SELECT u.user_id,
                       TRY_CAST(r.year_str AS SMALLINT) AS year
                FROM raw r
                JOIN "user" u ON r.user_raw_id = u.user_raw_id
                WHERE r.year_str != ''
            )
            SELECT DISTINCT user_id, year
            FROM mapped
            WHERE year IS NOT NULL
            ON CONFLICT DO NOTHING
        """
        try:
            self.conn.execute(sql)
            count = self.conn.execute("SELECT COUNT(*) FROM user_elite").fetchone()[0]
            logger.info(f"[Success] Inserted {count} rows into user_elite table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to bulk-insert user_elite: {e}")
            raise
