import duckdb
from pathlib import Path
from loguru import logger
from attrs import define, field

@define
class BusinessDatabaseCreator:
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

    def create_business_table(self):
        sequence_sql = '''
        CREATE SEQUENCE IF NOT EXISTS business_id START 1;
        '''
        create_sql = '''
        CREATE TABLE IF NOT EXISTS business (
            business_id INTEGER PRIMARY KEY DEFAULT nextval('business_id'),
            business_raw_id VARCHAR(255) UNIQUE NOT NULL,
            name TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            postal_code TEXT,
            latitude DOUBLE,
            longitude DOUBLE,
            stars DOUBLE,
            review_count INTEGER,
            is_open INTEGER
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created business table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create business table: {e}")
            raise

    def inert_business_data(self, data):
        try:
            insert_sql = '''
            INSERT INTO business (business_raw_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            self.conn.executemany(insert_sql, data)  # execute → executemany
            logger.info(f"[Success] Inserted {len(data)} rows into business table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to insert data into business table: {e}")
            raise

    def create_category_table(self):
        sequence_sql = '''
        CREATE SEQUENCE IF NOT EXISTS category_id_seq START 1;
        '''
        create_sql = '''
        CREATE TABLE IF NOT EXISTS category (
            category_id INTEGER PRIMARY KEY DEFAULT nextval('category_id_seq'),
            category TEXT UNIQUE
        );
        '''
        try:
            self.conn.execute(sequence_sql)
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created category table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create category table: {e}")
            raise

    def create_business_category_table(self):
        create_sql = '''
        CREATE TABLE IF NOT EXISTS business_category (
            business_id INTEGER REFERENCES business(business_id),
            category_id INTEGER REFERENCES category(category_id),
            PRIMARY KEY (business_id, category_id)
        );'''
        try:
            self.conn.execute(create_sql)
            logger.info(f"[Success] Created business_category table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create business_category table: {e}")
            raise

    def insert_categories(self, categories: list[str]):
        try:
            self.conn.executemany(
            "INSERT INTO category (category) VALUES (?) ON CONFLICT DO NOTHING",
            [(c,) for c in categories]
            )
            logger.info(f"[Success] Inserted categories batch: {len(categories)}")

        except Exception as e:
            logger.exception(f"[ERROR] insert_categories failed: {e}")
            raise

    def get_category_map(self, categories: list[str] = None) -> dict:
        try:
            if categories:
                placeholders = ",".join("?" * len(categories))
                rows = self.conn.execute(
                    f"SELECT category, category_id FROM category WHERE category IN ({placeholders})",
                    categories
                ).fetchall()
            else:
                rows = self.conn.execute("SELECT category, category_id FROM category").fetchall()
                logger.info(f"[Success] Retrieved category map with {len(rows)} entries")
            return {category: category_id for category, category_id in rows}
        except Exception as e:
            logger.exception(f"[ERROR] get_category_map failed: {e}")
            raise

    def insert_business_categories(self, data: list[tuple], category_map: dict):
        try:
            rows = [
                (business_id, category_map[category])
                for business_id, category in data
                if category in category_map
            ]
            self.conn.executemany(
                "INSERT INTO business_category (business_id, category_id) VALUES (?, ?) ON CONFLICT DO NOTHING",
                rows
            )
            logger.info(f"[Success] Inserted business-category relationships batch: {len(rows)}")
        except Exception as e:
            logger.exception(f"[ERROR] insert_business_categories failed: {e}")
            raise

    def create_business_hours_table(self):
        sql = '''
        CREATE TABLE IF NOT EXISTS business_hours (
            business_id INTEGER REFERENCES business(business_id),
            Monday_start_time TIME,
            Monday_end_time TIME,
            Tuesday_start_time TIME,
            Tuesday_end_time TIME,
            Wednesday_start_time TIME,
            Wednesday_end_time TIME,
            Thursday_start_time TIME,
            Thursday_end_time TIME,
            Friday_start_time TIME,
            Friday_end_time TIME,
            Saturday_start_time TIME,
            Saturday_end_time TIME,
            Sunday_start_time TIME,
            Sunday_end_time TIME);
        '''
        try:
            self.conn.execute(sql)
            logger.info(f"[Success] Created business_hours table in database")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to create business_hours table: {e}")
            raise   

    def insert_business_hours(self, data: list[tuple]):
        try:
            insert_sql = '''
            INSERT INTO business_hours (
                business_id, 
                Monday_start_time, Monday_end_time,
                Tuesday_start_time, Tuesday_end_time,
                Wednesday_start_time, Wednesday_end_time,
                Thursday_start_time, Thursday_end_time,
                Friday_start_time, Friday_end_time,
                Saturday_start_time, Saturday_end_time,
                Sunday_start_time, Sunday_end_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            self.conn.executemany(insert_sql, data)  # execute → executemany
            logger.info(f"[Success] Inserted {len(data)} rows into business_hours table")
        except Exception as e:
            logger.exception(f"[ERROR] Failed to insert data into business_hours table: {e}")
            raise

    def create_attribute_definition_table(self):
        # 表 1：ID 與屬性名稱對照
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS attribute_definition (
            attribute_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            value_type TEXT
        );
        CREATE SEQUENCE IF NOT EXISTS attribute_id_seq START 1;
        ''')

    def create_business_attribute_table(self):
        # 表 2：核心數據表
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS business_attribute (
            business_id INTEGER REFERENCES business(business_id),
            attribute_id INTEGER REFERENCES attribute_definition(attribute_id),
            value_text TEXT,
            value_bool BOOLEAN,
            value_num DOUBLE,
            value_json TEXT,
            sub_key TEXT,
            PRIMARY KEY (business_id, attribute_id, sub_key)
        )''')

    def insert_attribute_definition(self, data: list[tuple]):
        self.conn.executemany(
            "INSERT OR IGNORE INTO attribute_definition(attribute_id, name, value_type) VALUES (nextval('attribute_id_seq'), ?, ?)", 
            data
        )

    def insert_business_attributes(self, data: list[tuple]):
        self.conn.executemany(
            "INSERT INTO business_attribute VALUES (?, ?, ?, ?, ?, ?, ?)",
            data
        )

    def get_business_id_map(self, raw_ids: list[str]) -> dict:
        if not raw_ids:
            return {}
        placeholders = ",".join("?" * len(raw_ids))
        rows = self.conn.execute(
            f"SELECT business_raw_id, business_id FROM business WHERE business_raw_id IN ({placeholders})",
            raw_ids
        ).fetchall()
        return {raw_id: biz_id for raw_id, biz_id in rows}